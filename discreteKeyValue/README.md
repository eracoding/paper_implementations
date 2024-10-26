# Discrete Key-Value Bottleneck

This repository is custom implementation of the [paper](https://arxiv.org/pdf/2207.11240)

What is Discrete Key-Value Bottleneck?

This module learns to convert continuous input data into discrete codes by mapping inputs to a set of embeddings. These embeddings act like "keys" in a memory system, allowing the model to retrieve information efficiently, store important features, and discard redundant ones. This is particularly useful in applications like:

- Generative Models: Improve content generation by recalling relevant information stored in discrete embeddings.
- Knowledge Graphs: Efficiently encode and retrieve structured data.
- Representation Learning: Compress information without losing essential details.

## Usage
```
git clone https://github.com/your-repository/discrete-key-value-bottleneck.git
cd discrete-key-value-bottleneck
```

```
pip install -r requirements.txt
```
and run the notebook provided as example, or simply the following:

```
import torch
from discrete_key_value_bottleneck import DiscreteKeyValueBottleneck, DiscreteKeyValueBottleneckConfig

# Define the configuration
config = DiscreteKeyValueBottleneckConfig(
    dim=128,
    num_memories=256,
    num_memory_codebooks=4,
    average_pool_memories=True
)

# Instantiate the bottleneck model
bottleneck = DiscreteKeyValueBottleneck(config)

# Create an example input tensor
input_tensor = torch.randn(2, 64, 128)

# Forward pass through the bottleneck
output = bottleneck(input_tensor)

print(output.shape)

```

## Configuration
The DiscreteKeyValueBottleneck can be customized using the DiscreteKeyValueBottleneckConfig class. Here are the main configuration parameters:
```
dim: Input dimension size.
num_memories: Number of discrete memories (codebook size).
dim_embed: Dimension of the embeddings (optional).
num_memory_codebooks: Number of codebooks for memory.
dim_memory: Dimension of the memory values (optional).
average_pool_memories: Whether to apply average pooling to the retrieved memories.
encoder: Optional encoder module to process inputs before applying the bottleneck.
```