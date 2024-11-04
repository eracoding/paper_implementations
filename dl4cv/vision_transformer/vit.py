import torch
import torch.nn as nn

from utils import *


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MSA) layer for Transformer-based models.

    This layer splits the input into multiple attention heads, computes self-attention for each head, 
    and concatenates the results. It is commonly used in Transformer architectures for both NLP and 
    Vision Transformers (ViT).

    Args:
        dim_model (int): The dimensionality of the input and output representations.
        num_heads (int): The number of attention heads. Each head will have a dimensionality of `dim_model / num_heads`.

    Attributes:
        q_linear (nn.Linear): Linear layer to project input to query vectors.
        k_linear (nn.Linear): Linear layer to project input to key vectors.
        v_linear (nn.Linear): Linear layer to project input to value vectors.
        out_linear (nn.Linear): Final linear layer to project concatenated head outputs back to `dim_model`.
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int = 2,
    ):  
        
        super(MultiHeadSelfAttention, self).__init__()

        # Ensure the model dimension is divisible by the number of heads
        assert dim_model % num_heads == 0, f"dim_model {dim_model} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.dim_head = dim_model // num_heads

        # Linear layers to project input into query, key, and value spaces
        self.q = nn.Linear(dim_model, dim_model)
        self.k = nn.Linear(dim_model, dim_model)
        self.v = nn.Linear(dim_model, dim_model)

        # Output projection layer
        self.out_linear = nn.Linear(dim_model, dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head self-attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim_model), where:
                - batch_size is the number of input samples,
                - seq_len is the sequence length, and
                - dim_model is the dimensionality of each token in the sequence.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, dim_model) after multi-head self-attention is applied.
        """
        batch_size, seq_len, dim_model = x.size()

        # Project inputs to multi-head query, key, and value spaces
        q = self.q(x).view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # Apply softmax to get attention weights
        attended_values = torch.matmul(attention_weights, v)  # Shape: (batch_size, num_heads, seq_len, dim_head)
        
        # Concatenate heads and project to output dimension
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, dim_model)
        output = self.out_linear(attended_values)

        return output
    

class ViT(nn.Module):
    """
    Vision Transformer (ViT) implementation.

    This model divides an input image into patches, embeds them, adds positional encoding, 
    and applies Transformer layers to produce a classification output.

    Args:
        input_shape (tuple): Shape of the input image (channels, height, width).
        n_patches (int): Number of patches along each dimension.
        hidden_d (int): Dimensionality of the hidden layer.
        n_heads (int): Number of attention heads in multi-head self-attention.
        out_d (int): Dimensionality of the output (number of classes for classification).
    """

    def __init__(
        self,
        input_shape,
        n_patches=7,
        hidden_d=8,
        n_heads=2,
        out_d=10,
    ):
        super(ViT, self).__init__()

        # Input and patch size checks
        self.input_shape = input_shape
        self.n_patches = n_patches
        self.n_heads = n_heads

        assert input_shape[1] % n_patches == 0, "Input height not divisible by number of patches"
        assert input_shape[2] % n_patches == 0, "Input width not divisible by number of patches"

        self.patch_size = (input_shape[1] / n_patches, input_shape[2] / n_patches)
        self.hidden_d = hidden_d

        # 1) Linear layers for patch embeddings
        self.input_d = int(input_shape[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, hidden_d)

        # 2) Classification Token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding - forward layer

        # 4a) Layer Normalization 1
        self.ln1 = nn.LayerNorm((self.n_patches * self.n_patches + 1, self.hidden_d))

        # 4b) MSA and classification token
        self.msa = MultiHeadSelfAttention(self.hidden_d, n_heads)

        # 5a) Layer Normalization 2
        self.ln2 = nn.LayerNorm((self.n_patches * self.n_patches + 1, self.hidden_d))

        # 5b) Encoder MLP
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d),
            nn.ReLU()
        )

        # 6) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            images (Tensor): Batch of images of shape (batch_size, channels, height, width).
        
        Returns:
            Tensor: Output logits of shape (batch_size, out_d).
        """
        device = images.device

        # Divide images into patches and flatten each patch
        n, c, w, h = images.shape
        patches = images.reshape(n, self.n_patches ** 2, self.input_d)

        # Running linear layer for tokenization
        tokens = self.linear_mapper(patches).to(device)

        # Adding classification token to the tokens
        class_token = self.class_token.to(device)
        tokens = torch.stack([torch.vstack((class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        tokens += position_encoding(self.n_patches ** 2 + 1, self.hidden_d, device).repeat(n, 1, 1)

        # TRANSFORMER ENCODER BEGINS ###################################
        # NOTICE: MULTIPLE ENCODER BLOCKS CAN BE STACKED TOGETHER ######
        # Running Layer Normalization, MSA and residual connection
        tokens = tokens.to(device)
        ln1_tokens = self.ln1(tokens)
        msa_out = self.msa(ln1_tokens)
        out = tokens + msa_out

        # Running Layer Normalization, MLP and residual connection
        out = out + self.enc_mlp(self.ln2(out))
        # TRANSFORMER ENCODER ENDS   ###################################

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)
