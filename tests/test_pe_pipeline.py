import unittest
from pathlib import Path

import torch

from core import checkpoint
from core.config import expand_configs, to_dict
from core.mlflow_sink import _metric_values
from models.positional import Config as PositionalConfig
from models.positional import build_model
from tasks import get_task


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/pe_experiments"


class ConfigTest(unittest.TestCase):
    def test_kv_children_inherit_one_common_base(self):
        expected = {
            "nope": ("nope", "scratch"),
            "rope": ("rope", "scratch"),
            "wpe": ("wpe", "scratch"),
            "alibi": ("alibi", "scratch"),
            "rope_to_nope": ("nope", "checkpoint"),
            "nope_to_rope": ("rope", "checkpoint"),
        }
        for name, (encoding, init) in expected.items():
            config = expand_configs(CONFIG_ROOT / "kv_retrieval" / f"{name}.yaml")[0]
            self.assertEqual(config.model.pos_encoding, encoding)
            self.assertEqual(config.train.init, init)
            self.assertEqual(config.task.params["input_seq_len"], 64)
            self.assertEqual(config.train.data_seed, 424242)

    def test_c5_and_s5_use_current_task_parameters(self):
        c5 = expand_configs(CONFIG_ROOT / "c5/rope.yaml")[0]
        s5 = expand_configs(CONFIG_ROOT / "s5/rope.yaml")[0]
        self.assertEqual(set(c5.task.params), {"n_permutations"})
        self.assertEqual(set(s5.task.params), {"n_permutations"})
        recall = expand_configs(CONFIG_ROOT / "state_based_recall/rope.yaml")[0]
        self.assertEqual(set(recall.task.params), {"n_bits", "n_swaps", "max_bits"})

    def test_prepared_tasks_generate_and_collate(self):
        for task_name in ("kv_retrieval", "c5", "s5", "state_based_recall"):
            config = expand_configs(CONFIG_ROOT / task_name / "nope.yaml")[0]
            task = get_task(task_name, {**config.task.params, "seed": 1337})
            items = [task._sample_one() for _ in range(2)]
            x, y = task.collate(items, config.model.block_size)
            self.assertEqual(x.shape, y.shape)
            self.assertEqual(x.shape[0], 2)


class TransferTest(unittest.TestCase):
    def make_model(self, encoding):
        return build_model(PositionalConfig(
            vocab_size=64, block_size=16, n_layer=1, n_head=4,
            n_embd=32, pos_encoding=encoding,
        ))

    def populated_source(self, encoding="rope"):
        model = self.make_model(encoding)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))
        _, loss = model(tokens, targets)
        loss.backward()
        optimizer.step()
        return model, optimizer

    def test_rope_to_nope_preserves_all_weights_and_optimizer_moments(self):
        source, source_optimizer = self.populated_source("rope")
        saved = {
            "model": source.state_dict(),
            "optimizer_named": checkpoint.named_optimizer_state(source, source_optimizer),
        }
        target = self.make_model("nope")
        report = checkpoint.load_transfer_model_state(target, saved)
        self.assertFalse(report["skipped_source_positional"])
        self.assertFalse(report["new_target_positional"])
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])

        target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
        optimizer_report = checkpoint.restore_named_optimizer_state(
            target, target_optimizer, saved["optimizer_named"]
        )
        self.assertEqual(set(optimizer_report["restored"]),
                         {name for name, _ in target.named_parameters()})
        self.assertFalse(optimizer_report["skipped"])

    def test_wpe_removal_only_skips_positional_tensor(self):
        source, _ = self.populated_source("wpe")
        target = self.make_model("nope")
        report = checkpoint.load_transfer_model_state(target, {"model": source.state_dict()})
        self.assertEqual(report["skipped_source_positional"], ["transformer.wpe.weight"])

    def test_all_eve_pe_mechanisms_still_forward(self):
        for encoding in ("nope", "wpe", "rope", "fope", "alibi",
                         "relative_bias", "cope", "cape"):
            model = self.make_model(encoding)
            tokens = torch.randint(0, 64, (2, 16))
            logits, loss = model(tokens, tokens)
            self.assertEqual(logits.shape, (2, 16, 64))
            self.assertTrue(torch.isfinite(loss))

    def test_invalid_pe_hyperparameters_are_rejected(self):
        with self.assertRaises(ValueError):
            PositionalConfig(rope_theta=0.0)
        with self.assertRaises(ValueError):
            PositionalConfig(rpe_num_buckets=32, rpe_max_distance=16)
        with self.assertRaises(ValueError):
            PositionalConfig(fope_init_gain=-0.1)


class MLflowMetricTest(unittest.TestCase):
    def test_eval_and_summary_metrics_have_global_step(self):
        fields = {
            "iter": 17, "val_loss": 0.5, "val_acc": 0.75,
            "id_accuracy": 0.75, "evaluation_time_seconds": 1.2,
        }
        metrics = _metric_values("eval", fields)
        self.assertEqual(metrics["id_accuracy"], 0.75)
        self.assertEqual(metrics["evaluation_time_seconds"], 1.2)


if __name__ == "__main__":
    unittest.main()
