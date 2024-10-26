import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from vector_quantize_pytorch import VectorQuantize
from pydantic import BaseModel, validator, ConfigDict
from typing import Optional, Tuple, Union

# Helper functions

def exists(val) -> bool:
    return val is not None

def default(val, d):
    return val if exists(val) else d

# Configuration model using Pydantic

class DiscreteKeyValueBottleneckConfig(BaseModel):
    dim: int
    num_memories: int
    dim_embed: Optional[int] = None
    num_memory_codebooks: int = 1
    dim_memory: Optional[int] = None
    average_pool_memories: bool = True
    encoder: Optional[nn.Module] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types

    @validator("dim", "num_memories", "num_memory_codebooks", pre=True, always=True)
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
# Main Class

class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(self, config: DiscreteKeyValueBottleneckConfig, **kwargs):
        """
        Initializes the Discrete Key-Value Bottleneck.

        Args:
            config (DiscreteKeyValueBottleneckConfig): Configuration for the bottleneck.
            kwargs: Additional arguments for the VectorQuantize class.
        """
        super().__init__()
        self.config = config
        self.encoder = config.encoder
        self.dim_embed = default(config.dim_embed, config.dim)

        # Vector quantization initialization
        self.vq = VectorQuantize(
            dim=self.config.dim * self.config.num_memory_codebooks,
            codebook_size=self.config.num_memories,
            heads=self.config.num_memory_codebooks,
            separate_codebook_per_head=True,
            **kwargs
        )

        # Memory values
        dim_memory = default(config.dim_memory, config.dim)
        self.values = nn.Parameter(torch.randn(config.num_memory_codebooks, config.num_memories, dim_memory))

        # Random projection matrix
        rand_proj = torch.empty(config.num_memory_codebooks, self.dim_embed, config.dim)
        nn.init.xavier_normal_(rand_proj)
        self.register_buffer('rand_proj', rand_proj)

        self.average_pool_memories = config.average_pool_memories

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
        average_pool_memories: Optional[bool] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the Discrete Key-Value Bottleneck.

        Args:
            x (torch.Tensor): Input tensor.
            return_intermediates (bool): Whether to return intermediate results.
            average_pool_memories (Optional[bool]): Override the pooling configuration.
            kwargs: Additional arguments for the encoder.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]: Processed memory or tuple containing intermediate outputs.
        """
        average_pool_memories = default(average_pool_memories, self.average_pool_memories)

        # Encode if encoder is present
        if exists(self.encoder):
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, **kwargs)
                x.detach_()

        assert x.shape[-1] == self.dim_embed, (
            f"Encoding has a dimension of {x.shape[-1]} but expected {self.dim_embed}"
        )

        # Apply random projections and vector quantization
        x = einsum('b n d, c d e -> b n c e', x, self.rand_proj)
        x = rearrange(x, 'b n c e -> b n (c e)')
        vq_out = self.vq(x)

        quantized, memory_indices, commit_loss = vq_out

        # Ensure correct shape for memory indices
        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '... -> ... 1')

        memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        # Gather memory values based on quantization indices
        values = repeat(self.values, 'h n d -> b h n d', b=memory_indices.shape[0])
        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d=values.shape[-1])
        memories = values.gather(2, memory_indices)

        # Average pooling if required
        if average_pool_memories:
            memories = reduce(memories, 'b h n d -> b n d', 'mean')

        # Return intermediate results if specified
        if return_intermediates:
            return memories, (quantized, memory_indices, commit_loss)

        return memories

# Example usage
# config = DiscreteKeyValueBottleneckConfig(
#     dim=128,
#     num_memories=256,
#     num_memory_codebooks=4,
#     average_pool_memories=True
# )

# bottleneck = DiscreteKeyValueBottleneck(config)
# input_tensor = torch.randn(2, 64, 128)  # Example input
# output = bottleneck(input_tensor)
