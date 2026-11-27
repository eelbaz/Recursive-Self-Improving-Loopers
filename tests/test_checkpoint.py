"""Unit & round-trip tests for rse_looper.checkpoint."""

import json
import os
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rse_looper.checkpoint import (
    apply_weight_deltas,
    export_edited_checkpoint,
    is_editable_param,
    normalize_param_name,
    restore_weights,
    transformers_name_to_ckpt_deltas,
)


def test_normalize_and_allowlist_blocks_experts_vision_embeds():
    assert normalize_param_name("layers.6.self_attn.o_proj.linear.weight") == "model.layers.6.self_attn.o_proj.linear.weight"
    assert normalize_param_name("final_input.query") == "model.final_input.query"
    assert is_editable_param("model.layers.17.self_attn.o_proj.linear.weight")
    assert is_editable_param("layers.11.mlp.gate.e_score_correction_bias")
    assert is_editable_param("model.layers.6.mlp_message_gate.down_proj.weight")
    assert is_editable_param("model.message_input.query")
    assert is_editable_param("model.final_input.query")

    # Strictly forbidden tensors (quantized experts, vision tower, embeddings, lm_head, scales)
    for forbidden in [
        "model.layers.6.mlp.experts.0.down_proj.weight",
        "model.visual.backbone.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.6.mlp.experts.weight_scale",
        "model.layers.0.input_layernorm.weight",
    ]:
        assert not is_editable_param(forbidden)
        with pytest.raises(ValueError):
            transformers_name_to_ckpt_deltas(forbidden, torch.zeros(2, 2))


def test_inverse_mapping_offsets_and_transposes():
    # o_proj.linear.weight transposes into wo_ud[3072:6144]
    d_o = torch.randn(2304, 3072)
    [(ck, sl, d)] = transformers_name_to_ckpt_deltas("model.layers.17.self_attn.o_proj.linear.weight", d_o)
    assert ck == "model.llm.layers.17.attn.wo_ud.weight"
    assert sl == slice(3072, 6144)
    assert torch.allclose(d, d_o.T)

    # MessageGate uses offset lidx - 6
    d_mg = torch.randn(2304, 64)
    [(ck, sl, d)] = transformers_name_to_ckpt_deltas("model.layers.10.mlp_message_gate.message_gate_proj.weight", d_mg)
    assert ck == "model.llm.middle_cycle_cross_pass_gates.4.mlp_gate.up_proj.weight"
    assert sl == slice(2304, 4608)

    # MessageGate rejected outside middle loop [6..17]
    with pytest.raises(ValueError):
        transformers_name_to_ckpt_deltas("model.layers.3.mlp_message_gate.down_proj.weight", torch.zeros(64, 4608))

    # final_input.query maps to attn_res layer 24
    [(ck, sl, _)] = transformers_name_to_ckpt_deltas("model.final_input.query", torch.zeros(2304))
    assert ck == "model.llm.attn_res.layers.24.attention_input.query"
    assert sl is None


def test_export_edited_checkpoint_zero_drift_roundtrip(tmp_path):
    base_dir = tmp_path / "base"
    out_dir = tmp_path / "edited"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))

    # Build minimal synthetic safetensors containing BF16 editable + uint8 quantized expert + vision tensors
    orig_tensors = {
        "model.llm.layers.17.attn_sconv.conv.weight": torch.ones(16, 1, 4, dtype=torch.bfloat16),
        "model.llm.attn_res.layers.24.attention_input.query": torch.ones(16, dtype=torch.float32),
        "model.llm.layers.17.moe.experts.0.w1_nvfp4": torch.randint(0, 255, (32, 32), dtype=torch.uint8),
        "model.visual.patch_embed.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    }
    save_file(orig_tensors, str(base_dir / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})

    deltas = {
        "model.layers.17.post_attention_convolution.conv1d.weight": torch.full((16, 1, 4), 0.05),
        "model.final_input.query": torch.full((16,), -0.02),
    }
    report = export_edited_checkpoint(str(base_dir), str(out_dir), deltas)
    assert len(report) == 2

    with safe_open(str(out_dir / "model-00001-of-00001.safetensors"), framework="pt", device="cpu") as f:
        new_keys = set(f.keys())
        assert new_keys == set(orig_tensors.keys())
        # Untouched quantized expert & vision tensors MUST be bit-for-bit identical!
        assert torch.equal(f.get_tensor("model.llm.layers.17.moe.experts.0.w1_nvfp4"), orig_tensors["model.llm.layers.17.moe.experts.0.w1_nvfp4"])
        assert torch.equal(f.get_tensor("model.visual.patch_embed.weight"), orig_tensors["model.visual.patch_embed.weight"])
        # Edited tensors match expected values & preserve dtypes
        ed_q = f.get_tensor("model.llm.attn_res.layers.24.attention_input.query")
        assert ed_q.dtype == torch.float32
        assert torch.allclose(ed_q, torch.full((16,), 0.98), atol=1e-6)
        # Parameter count strictly identical
        orig_params = sum(t.numel() for t in orig_tensors.values())
        new_params = sum(f.get_tensor(k).numel() for k in new_keys)
        assert orig_params == new_params


def test_apply_and_restore_weights_exact():
    m = torch.nn.Linear(8, 4, bias=False)
    orig = m.weight.detach().clone()
    backups = apply_weight_deltas(m, {"weight": torch.full((4, 8), 0.1)})
    assert torch.allclose(m.weight, orig + 0.1, atol=1e-6)
    restore_weights(m, backups)
    assert torch.equal(m.weight, orig)
