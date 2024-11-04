import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class AttentionHead(nn.Module):
    #def __init__(self, dim_in: int, dim_q: int, dim_k: int): # mistake, from the paper it says dim_q = dim_k but not dim_k != dim_v
    def __init__(self, dim_in: int, dim_qk: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_qk)
        self.k = nn.Linear(dim_in, dim_qk)
        self.v = nn.Linear(dim_in, dim_v)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        return self.scaled_dot_product_attention(q, k, v, mask)

    @staticmethod
    def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computed scaled dot product of attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, dim_qk).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, dim_qk).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, dim_v).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor after attention is applied.
        """
        # Scaled dot-product
        scores = q.bmm(k.transpose(1, 2)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32) + 1e-8)

        # Apply mask if needed
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax across the last dim
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the output
        output = attention_weights.bmm(v)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_qk: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_qk, dim_v) for _ in range(num_heads)]
        )

        self.linear = nn.Linear(num_heads * dim_qk, dim_in)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        return self.linear(
            torch.cat([h(query, key, value, mask) for h in self.heads], dim=-1)
        )


class ResidualConnections(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        """
        Residual connection with layer normalization and optional dropout.
        
        Args:
            sublayer (nn.Module): The sublayer to be applied, e.g., MultiHeadAttention or FeedForward.
            dimension (int): The dimension for the LayerNorm.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual layer.
        
        Assumes that the first tensor in `tensors` is the primary input for the residual connection.
        
        Args:
            *tensors: Input tensors where the first is assumed to be the main residual input.
        
        Returns:
            Tensor: The output tensor after applying residual connection, dropout, and normalization.
        """
        # Apply sublayer, then add the residual connection, followed by normalization
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))
        


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Transformer Encoder Layer with residual connections around multi-head
        attention and feed-forward sub-layers.

        Args:
            dim_model (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feed-forward layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # Calculate dimensions for query/key and value based on number of heads
        dim_qk = dim_v = max(dim_model // num_heads, 1)
        
        # Multi-head attention layer with residual connection 
        self.attention = ResidualConnections(
            MultiHeadAttention(num_heads, dim_model, dim_qk, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )

        # Feed-forward network with residual connection
        self.feed_forward = ResidualConnections(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout
        )

    def forward(self, src):
        """
        Forward pass through the Transformer Encoder Layer.
        
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, dim_model).
        
        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True
    ):
        """
        Transformer Encoder comprising multiple encoder layers.

        Args:
            num_layers (int): Number of encoder layers.
            dim_model (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feed-forward layer.
            dropout (float): Dropout probability.
            apply_positional_encoding (bool): Whether to add positional encoding.
        """
        super().__init__()
        self.apply_positional_encoding = apply_positional_encoding

        # Stack of encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def add_positional_encoding(self, src: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, dim_model).
        
        Returns:
            Tensor: Input tensor with positional encoding added.
        """
        seq_len, dimension = src.size(1), src.size(2)
        return src + position_encoding(seq_len, dimension, device=src.device)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Encoder.

        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, dim_model).

        Returns:
            Tensor: Encoded tensor of the same shape as input.
        """
        # Optionally add positional encoding
        if self.apply_positional_encoding:
            src = self.add_positional_encoding(src)

        # Pass through each encoder layer
        for layer in self.layers:
            src = layer(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Transformer Decoder Layer with two residual-attention connections and a 
        feed-forward network, each wrapped in residual connections.

        Args:
            dim_model (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feed-forward layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # Dimensions for query/key and value
        dim_qk = dim_v = max(dim_model // num_heads, 1)

        # Self-attention with residual connection
        self.attention1 = ResidualConnections(
            MultiHeadAttention(num_heads, dim_model, dim_qk, dim_v),
            dimension=dim_model,
            dropout=dropout
        )

        # Cross-attention with residual connection
        self.attention2 = ResidualConnections(
            MultiHeadAttention(num_heads, dim_model, dim_qk, dim_v),
            dimension=dim_model,
            dropout=dropout
        )

        # Feed-forward with residual connection
        self.feed_forward = ResidualConnections(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout
        )
    
    def forward(self, trg: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Decoder Layer.
        
        Args:
            trg (Tensor): Target tensor of shape (batch_size, seq_len, dim_model).
            memory (Tensor): Memory tensor from the encoder of shape (batch_size, seq_len, dim_model).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, dim_model).
        """
        trg = self.attention1(trg, trg, trg) # Self-attention
        trg = self.attention2(trg, memory, memory) # Cross-attention with encoder output
        return self.feed_forward(trg)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True
    ):
        """
        Transformer Decoder consisting of multiple decoder layers.

        Args:
            num_layers (int): Number of decoder layers.
            dim_model (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feed-forward layer.
            dropout (float): Dropout probability.
            apply_positional_encoding (bool): Whether to add positional encoding.
        """
        super().__init__()
        self.apply_positional_encoding = apply_positional_encoding

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final linear layer to project to output vocabulary size or target dimension
        self.linear = nn.Linear(dim_model, dim_model)
    
    def add_positional_encoding(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the target tensor.
        
        Args:
            trg (Tensor): Target tensor of shape (batch_size, seq_len, dim_model).
        
        Returns:
            Tensor: Target tensor with positional encoding added.
        """
        seq_len, dimension = trg.size(1), trg.size(2)
        return trg + position_encoding(seq_len, dimension, device=trg.device)

    def forward(self, trg: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Decoder.
        
        Args:
            trg (Tensor): Target tensor of shape (batch_size, seq_len, dim_model).
            memory (Tensor): Memory tensor from the encoder of shape (batch_size, seq_len, dim_model).
        
        Returns:
            Tensor: Decoded tensor of shape (batch_size, seq_len, dim_model).
        """
        # Optionally add positional encoding
        if self.apply_positional_encoding:
            trg = self.add_positional_encoding(trg)

        # Pass through each decoder layer
        for layer in self.layers:
            trg = layer(trg, memory)
        
        # Linear projection, softmax should be applied externally if needed
        # return torch.softmax(self.linear(trg), dim=-1) #The softmax should typically be applied outside the decoder in a final output layer if necessary.
        return self.linear(trg)


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Transformer model consisting of an encoder and decoder.

        Args:
            num_encoder_layers (int): Number of layers in the encoder.
            num_decoder_layers (int): Number of layers in the decoder.
            dim_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers, 
            dim_model=dim_model, 
            num_heads=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    
    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model.

        Args:
            src (Tensor): Source tensor of shape (batch_size, src_seq_len, dim_model).
            trg (Tensor): Target tensor of shape (batch_size, trg_seq_len, dim_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, trg_seq_len, dim_model).
        """
        # Pass through encoder
        memory = self.encoder(src)

        # Pass through decoder with the encoder's output as memory
        output = self.decoder(trg, memory)

        return output


# src = torch.rand(64, 32, 512)
# tgt = torch.rand(64, 16, 512)
# out = Transformer()(src, tgt)
# print(out.shape)
# # torch.Size([64, 16, 512])