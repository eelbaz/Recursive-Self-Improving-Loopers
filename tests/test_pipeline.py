"""Unit tests for the production autonomous pipeline CLI (rse_looper.pipeline)."""

import math
import pytest
import torch

from rse_looper.pipeline import PipelineConfig, main, merge_generation_deltas, parse_config


def test_parse_config_and_validation():
    cfg, dry_run = parse_config(["--stage", "fpsolve", "--gpus", "0,1", "--dry-run"])
    assert cfg.stage == "fpsolve"
    assert cfg.gpus == "0,1"
    assert cfg.resume is True
    assert dry_run is True

    with pytest.raises(ValueError):
        PipelineConfig(stage="invalid").validate()
    with pytest.raises(ValueError):
        PipelineConfig(fpsolve_max_rel=0.9).validate()
    with pytest.raises(ValueError):
        PipelineConfig(gspo_max_gens=0).validate()


def test_merge_generation_deltas_weighted_average_and_union_keys():
    d1 = {"a": torch.tensor([2.0, 4.0]), "b": torch.tensor([1.0])}
    d2 = {"a": torch.tensor([6.0, 8.0]), "c": torch.tensor([3.0])}
    m = merge_generation_deltas([d1, d2], weights=[1.0, 1.0])
    assert torch.allclose(m["a"], torch.tensor([4.0, 6.0]))
    assert math.isclose(m["b"].item(), 0.5, rel_tol=1e-6)
    assert math.isclose(m["c"].item(), 1.5, rel_tol=1e-6)

    with pytest.raises(ValueError):
        merge_generation_deltas([])
    with pytest.raises(ValueError):
        merge_generation_deltas([d1, d2], weights=[1.0])


def test_pipeline_dry_run_executes_preflight_cleanly():
    res = main(["--stage", "both", "--gpus", "0,1,2,3,4,5", "--dry-run"])
    assert res["dry_run"] is True
    assert res["stage"] == "both"
    assert res["gpus"] == [0, 1, 2, 3, 4, 5]
    assert "causal_conv1d_installed" in res["preflight"]


def test_stage_modules_expose_runnable_cli_parsers():
    from rse_looper.fpsolve import build_parser as fp_parser
    from rse_looper.gspo import main as gspo_main

    fp_args = fp_parser().parse_args(["--lam-keep", "4.0", "--max-rel", "0.25"])
    assert fp_args.lam_keep == 4.0 and fp_args.max_rel == 0.25

    gspo_res = gspo_main(["--gpus", "0,1", "--dry-run"])
    assert gspo_res["gpus"] == [0, 1]
