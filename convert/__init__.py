"""Convert mlx-speech checkpoints to Core AI models. One subpackage per family."""

from convert.workspace import confine

# Before any submodule imports torch.
confine()
