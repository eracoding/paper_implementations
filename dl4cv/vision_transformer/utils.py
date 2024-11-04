import torch
import torch.nn as nn


def position_encoding(seq_len: int, dim_model: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generates sinusoidal positional encodings.

    Args:
        seq_len (int): The length of the sequence.
        dim_model (int): The dimensionality of the model (must be even for sin-cos pairing).
        device (torch.device): The device on which to create the encoding tensor.
    
    Returns:
        Tensor: The positional encoding tensor of shape (1, seq_len, dim_model)
    """
    # Position indices for each position in the sequence (shape: [1, seq_len, 1])
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).reshape(1, -1, 1)

    # Dimension indices for each encoding dimension (shape: [1, 1, dim_model])
    dim = torch.arange(dim_model, dtype=torch.float32, device=device).reshape(1, 1, -1)

    # Compute the phase values based on position and dimension indices
    # This uses `dim_model` to scale each dimension in the sinusoidal pattern
    phase = pos / 1e4 ** (dim / dim_model)

    # Apply sin to even dimensions and cos to odd dimensions
    encoding = torch.zeros((1, seq_len, dim_model), device=device)
    encoding[..., 0::2] = torch.sin(phase[..., 0::2]) # Apply sin to even dimensions
    encoding[..., 1::2] = torch.cos(phase[..., 1::2]) # Apply cos to odd dimensions

    return encoding
    

def feed_forward(dim_model: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    """
    Fully connected feed-forward network, which is applied to each position separately and identically.

    Args:
        dim_model (int): The dimensionality of the model.
        dim_feedforward (int): inner-layer dimensionality.
    
    Returns:
        nn.Module: Sequential layer of feed-forward neural network.
    """
    return nn.Sequential(
        nn.Linear(dim_model, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_model),
    ) # Possible improvements from vanila ViT is adding dropout, layer norm, and etc.