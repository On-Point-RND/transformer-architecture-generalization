import torch
from model import (
    AttentionConfig, FFNConfig, ModelConfig, LoopConfig,
    AlgorithmicTransformer, LoopedTransformer,
    compute_iso_param_hidden_dim,
)

torch.manual_seed(0)
D_MODEL, VOCAB, BATCH, SEQ = 128, 64, 3, 24
input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))


def make_config(pe_kind, attn_kind, pattern, softmax_kind, ffn_kind):
    if attn_kind == "mha":
        n_heads, n_kv_heads, share_kv = 4, 4, False
    elif attn_kind == "gqa":
        n_heads, n_kv_heads, share_kv = 4, 2, False
    elif attn_kind == "mqa":
        n_heads, n_kv_heads, share_kv = 4, 1, False
    elif attn_kind == "mqa_kv":
        n_heads, n_kv_heads, share_kv = 4, 1, True

    attn_cfg = AttentionConfig(
        n_heads=n_heads, n_kv_heads=n_kv_heads, share_kv=share_kv,
        pattern=pattern, window_size=8 if pattern == "local" else None,
        n_global_tokens=2 if pattern == "local" else 0,
        softmax_kind=softmax_kind,
    )
    hidden = compute_iso_param_hidden_dim(ffn_kind, D_MODEL, target_params=2 * D_MODEL * 256)
    ffn_cfg = FFNConfig(kind=ffn_kind, hidden_dim=hidden)
    return ModelConfig(
        d_model=D_MODEL, n_layers=2, vocab_size=VOCAB,
        attention=attn_cfg, ffn=ffn_cfg, positional_encoding=pe_kind,
    )


def test_all_positional_encodings():
    n_ok = 0
    for pe_kind in ["learned_absolute", "sinusoidal", "rope", "alibi", "nope"]:
        config = make_config(pe_kind, "mha", "full", "standard", "relu")
        model = AlgorithmicTransformer(config)
        logits = model(input_ids)
        assert logits.shape == (BATCH, SEQ, VOCAB), (pe_kind, logits.shape)
        assert torch.isfinite(logits).all(), f"non-finite logits for {pe_kind}"
        n_ok += 1
    print(f"OK all {n_ok} positional encodings forward-pass cleanly")


def test_all_attention_variants_x_patterns_x_softmax():
    n_ok = 0
    for attn_kind in ["mha", "gqa", "mqa", "mqa_kv"]:
        for pattern in ["full", "local"]:
            for softmax_kind in ["standard", "taylor"]:
                config = make_config("rope", attn_kind, pattern, softmax_kind, "relu")
                model = AlgorithmicTransformer(config)
                logits = model(input_ids)
                assert logits.shape == (BATCH, SEQ, VOCAB)
                assert torch.isfinite(logits).all(), (attn_kind, pattern, softmax_kind)
                n_ok += 1
    print(f"OK all {n_ok} attention x pattern x softmax combinations forward-pass cleanly")


def test_ffn_variants():
    for ffn_kind in ["relu", "swiglu"]:
        config = make_config("learned_absolute", "mha", "full", "standard", ffn_kind)
        model = AlgorithmicTransformer(config)
        logits = model(input_ids)
        assert logits.shape == (BATCH, SEQ, VOCAB)
    print("OK ReLU and SwiGLU FFN both forward-pass cleanly")


def test_local_pattern_actually_restricts_attention():
    """Sanity check that local attention really can't see far tokens:
    gradient of output at position T-1 w.r.t. the embedding at position
    0 should be exactly zero under a local pattern with no global tokens
    and a window smaller than the sequence, but non-zero under full attention.
    """
    config_full = make_config("nope", "mha", "full", "standard", "relu")
    config_local = make_config("nope", "mha", "local", "standard", "relu")
    config_local.attention.n_global_tokens = 0  # no sink tokens for this check

    for config, expect_zero_grad in [(config_full, False), (config_local, True)]:
        model = AlgorithmicTransformer(config)
        emb = model.token_emb(input_ids).detach().clone().requires_grad_(True)
        positions = model._positions(input_ids)
        x = model.pos_enc.embed(emb, positions)
        for block in model.blocks:
            x = block(x, positions)
        logits = model.lm_head(model.norm_out(x))
        loss = logits[0, -1].sum()
        loss.backward()
        grad_at_pos0 = emb.grad[0, 0].abs().sum().item()
        if expect_zero_grad:
            assert grad_at_pos0 < 1e-6, f"local attention leaked info from position 0: grad={grad_at_pos0}"
        else:
            assert grad_at_pos0 > 1e-6, "full attention should see position 0"
    print("OK local attention pattern verified to actually restrict the receptive field")


def test_looped_transformer():
    config = make_config("rope", "gqa", "full", "standard", "swiglu")
    loop_cfg = LoopConfig(k_iters=6)
    model = LoopedTransformer(config, loop_cfg)
    logits = model(input_ids)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert torch.isfinite(logits).all()

    acts = model.get_iteration_activations(input_ids, step=2)
    assert acts.shape == (BATCH, SEQ, D_MODEL)

    # sanity: block is genuinely shared -- same object reused every step,
    # not len(k_iters) separate copies
    assert sum(p.numel() for p in model.block.parameters()) == \
           sum(p.numel() for p in model.parameters()) - model.timestep_emb.weight.numel() \
           - model.token_emb.weight.numel() - model.lm_head.weight.numel() \
           - sum(p.numel() for p in model.norm_out.parameters())
    print("OK looped transformer forward-pass, iteration snapshot, and weight-sharing verified")


if __name__ == "__main__":
    test_all_positional_encodings()
    test_all_attention_variants_x_patterns_x_softmax()
    test_ffn_variants()
    test_local_pattern_actually_restricts_attention()
    test_looped_transformer()
    print("\nAll project-model smoke tests passed.")
