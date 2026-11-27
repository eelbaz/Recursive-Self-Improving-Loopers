"""Given/When/Then contracts for lightweight package access."""

import subprocess
import sys


def test_given_no_training_runtime_when_importing_grading_then_no_heavy_imports():
    """Given an evidence-only process, When grading, Then no ML runtime is imported."""
    # Given: an environment deliberately denying training-only dependencies.
    script = r'''
import importlib.abc
import sys

class NoTrainingRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "safetensors", "transformers"}:
            raise ImportError("Training runtime must not load for evidence-only work: " + fullname)

sys.meta_path.insert(0, NoTrainingRuntime())

# When: users import the package and its public grading helper.
import rse_looper
from rse_looper import grade_math_output
result = grade_math_output(r"<|content_text|>\boxed{8}<|message_end|>", "8")

# Then: grading works without importing the model/training stack.
assert result["correct"] is True
assert not {"torch", "safetensors", "transformers"}.intersection(sys.modules)
'''
    # Then: the supported public import works in a fresh interpreter.
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_given_unknown_public_attribute_when_accessed_then_attribute_error():
    """Given a package, When an unknown public name is requested, Then fail clearly."""
    import rse_looper
    import pytest

    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(rse_looper, "does_not_exist")
