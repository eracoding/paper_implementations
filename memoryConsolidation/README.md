# A Generative Model of Memory Construction and Consolidation

The repository includes custom implementation of the code for modeling consolidation using a teacher-student learning approach, where initial memory representations are replayed to train a generative model. It contains only one notebook which is implementation of memory consolidation.

[Paper - Generative Model of Memory Construction and Consolidation](https://www.nature.com/articles/s41562-023-01799-z)

# Installation

```
pip install -r requirements.txt
```

Or if you are using poetry:
```
poetry install
```

Then run the notebook.

# Some insights that was implemented:

The work detailed in this implementation includes the following:

Developing a framework to encode, decode, and recall memories using modern Hopfield networks and variational autoencoders.
Designing and training a VAE to learn meaningful latent representations from encoded memories, assessing its performance with multiple tests, including accuracy, error metrics, and visualizations.
Extending the standard memory encoding model to incorporate both sensory and conceptual components, enabling more robust and flexible memory storage and retrieval.
Testing various scenarios, including using pre-trained weights, examining interpolation abilities, and adjusting error thresholds to observe the effects on the model's performance.
Providing a comprehensive set of outputs, from graphs and visualizations to reports summarizing training outcomes and overall results.

# References:

1. Inspired by the original [paper implementation](https://github.com/ellie-as/generative-memory) which is writtent in tensorflow. I rewrote most parts with some modifications in pytorch.
2. Hopfield network implementation was inspired by https://github.com/ml-jku/hopfield-layers (added some modifications), [paper](https://arxiv.org/abs/2008.02217).