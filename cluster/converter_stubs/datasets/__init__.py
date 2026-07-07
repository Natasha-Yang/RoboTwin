"""Minimal optional datasets shim for MolmoAct checkpoint conversion.

The converter imports ``olmo.util``, which imports this one function even
though checkpoint conversion does not load Hugging Face datasets.
"""


def disable_progress_bar() -> None:
    pass
