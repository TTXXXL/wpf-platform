"""Neural forecasters sharing one training loop (trainer.py).

Network modules that import torch are imported lazily inside ``build_network`` so that
the model registry works without torch installed.
"""

from .trainer import NetworkSpec, NeuralForecaster

__all__ = ["NetworkSpec", "NeuralForecaster"]
