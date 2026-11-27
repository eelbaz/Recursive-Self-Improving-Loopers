"""Local loop-dynamics editing, self-training, and experiment evidence tools.

Public helpers are imported on first access. Inspecting evidence or grading
answers must not initialize a training runtime or pre-import CLI modules.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "LoopConfig": "looping",
    "PipelineConfig": "pipeline",
    "accept_3regime_rsi_step": "gspo",
    "clip_generation_deltas": "gspo",
    "compute_loop_advantage_gspo_groups": "gspo",
    "detect_repetition": "grader",
    "enable_fast_math": "preflight",
    "export_edited_checkpoint": "checkpoint",
    "extract_boxed": "grader",
    "grade_math_output": "grader",
    "gspo_dapo_sequence_loss": "gspo",
    "install_midstack_loop_forward": "looping",
    "is_editable_param": "checkpoint",
    "merge_generation_deltas": "pipeline",
    "normalize_param_name": "checkpoint",
    "preflight_gpu_check": "preflight",
    "solve_nullspace_ridge": "fpsolve",
    "split_inkling_output": "grader",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
