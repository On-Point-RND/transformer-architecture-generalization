import pytest
import torch

from models.positional import Config, Model


def make_model(mode, block_size=12):
    return Model(Config(
        pos_encoding=mode,
        block_size=block_size,
        vocab_size=17,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    ))


@pytest.mark.parametrize("mode", ["abs_shift", "pose"])
def test_shifted_modes_eval_are_deterministic_and_offset_free(mode):
    model = make_model(mode).eval()
    seen = []
    handle = model.transformer.wpe.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    x = torch.randint(0, model.config.vocab_size, (3, 7))

    first, _ = model(x)
    second, _ = model(x)
    handle.remove()

    assert torch.equal(first, second)
    assert len(seen) == 2
    assert all(torch.equal(pos, torch.arange(x.size(1))) for pos in seen)


@pytest.mark.parametrize("mode", ["abs_shift", "pose"])
def test_shifted_modes_are_safe_at_max_length(mode):
    model = make_model(mode, block_size=8).train()
    seen = []
    handle = model.transformer.wpe.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    x = torch.randint(0, model.config.vocab_size, (2, 8))

    model(x)
    handle.remove()

    expected = torch.arange(8).expand(2, 8)
    assert torch.equal(seen[0], expected)


def test_shifted_modes_have_same_parameter_count_as_wpe():
    counts = {
        mode: sum(parameter.numel() for parameter in make_model(mode).parameters())
        for mode in ("wpe", "abs_shift", "pose")
    }
    assert counts["abs_shift"] == counts["wpe"]
    assert counts["pose"] == counts["wpe"]


def test_pose_samples_monotonic_in_range_positions():
    model = make_model("pose", block_size=16)
    positions = model.sample_pose_positions(64, 7, "cpu")

    assert positions.shape == (64, 7)
    assert positions.dtype == torch.long
    assert int(positions.min()) >= 0
    assert int(positions.max()) < model.config.block_size
    assert bool((positions[:, 1:] > positions[:, :-1]).all())


def test_explicit_positions_are_used_and_validated():
    model = make_model("pose", block_size=8).train()
    seen = []
    handle = model.transformer.wpe.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    x = torch.randint(0, model.config.vocab_size, (2, 4))
    positions = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]])

    model(x, positions=positions)
    handle.remove()

    assert torch.equal(seen[0], positions)
    with pytest.raises(ValueError, match="shape"):
        model(x, positions=positions[:, :-1])


def test_one_dimensional_explicit_positions_broadcast_across_batch():
    model = make_model("pose", block_size=8).train()
    seen = []
    handle = model.transformer.wpe.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    x = torch.randint(0, model.config.vocab_size, (2, 4))
    positions = torch.tensor([0, 2, 4, 6])

    model(x, positions=positions)
    handle.remove()

    assert torch.equal(seen[0], positions)


def test_shift_and_pose_samples_are_not_plain_positions():
    shifted = make_model("abs_shift", block_size=16).train()
    x = torch.randint(0, shifted.config.vocab_size, (64, 7))
    targets = torch.full_like(x, -1)
    targets[:, 5:] = torch.tensor([1, 2])
    seen = []
    handle = shifted.transformer.wpe.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    shifted(x, targets)
    handle.remove()
    assert bool((seen[0][:, 0] > 0).any())

    pose = make_model("pose", block_size=16)
    positions = pose.sample_pose_positions(128, 7, "cpu")
    gaps = positions[:, 1:] - positions[:, :-1]
    assert bool((gaps > 1).any())


def test_generate_temporarily_uses_eval_positions_and_restores_training_mode():
    model = make_model("pose", block_size=8).train()
    seen = []
    handle = model.transformer.wpe.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    prompt = torch.randint(0, model.config.vocab_size, (2, 3))

    generated = model.generate(prompt, max_new_tokens=2, top_k=1)
    handle.remove()

    assert generated.shape == (2, 5)
    assert model.training
    assert all(position.ndim == 1 for position in seen)


@pytest.mark.parametrize("mode", ["abs_shift", "pose"])
def test_training_positions_use_each_rows_effective_length(mode):
    model = make_model(mode, block_size=8).train()
    seen = []
    handle = model.transformer.wpe.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.full_like(x, -1)
    targets[0, 2:4] = torch.tensor([1, 2])  # valid x span ends at index 3
    targets[1, 4:6] = torch.tensor([3, 4])  # valid x span ends at index 5

    model(x, targets)
    handle.remove()

    positions = seen[0]
    assert positions.shape == x.shape
    assert torch.equal(positions[0, 4:], torch.zeros(4, dtype=torch.long))
    assert torch.equal(positions[1, 6:], torch.zeros(2, dtype=torch.long))
    assert bool((positions[0, 1:4] > positions[0, :3]).all())
    assert bool((positions[1, 1:6] > positions[1, :5]).all())
    assert int(positions.max()) < model.config.block_size
