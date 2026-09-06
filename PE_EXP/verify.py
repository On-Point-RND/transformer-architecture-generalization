"""Protocol checks plus optional CUDA FP16 checks; artifacts stay in results/verification."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run as runner
import numpy as np
import torch
from core import checkpoint
from core.config import OptimizerConfig
from models.positional import Model, Config, SLOTS
from tasks import get_task

runner.torch, runner.np, runner.checkpoint = torch, np, checkpoint
runner.Model, runner.ModelConfig, runner.OptimizerConfig = Model, Config, OptimizerConfig
runner.get_task, runner.SLOTS = get_task, SLOTS
torch.set_num_threads(1)


class ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for FP16")
    def test_all_encodings_cuda_fp16(self):
        with patch.object(runner, "precision", return_value=torch.float16):
            for encoding in runner.read_json(runner.HERE / "experiment.json")["encodings"]:
                with self.subTest(encoding=encoding):
                    task = get_task("relative_offset_copy", {"seed": 7, "length_range": (256, 256)})
                    model = Model(Config(**self.cfg["model"], vocab_size=task.vocab_size,
                                         pos_encoding=encoding)).to("cuda:0")
                    x, y = runner.batch(task, task.sample(2), "cuda:0")
                    with runner.autocast("cuda:0"):
                        _, loss = model(x, y)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                        if p.grad is not None))
                    result = runner.evaluate(model, task, task.sample(2), "cuda:0", 1)
                    self.assertEqual(result["n"], 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for FP16 scaling")
    def test_cuda_fp16_overflow_retry_and_resume(self):
        # Force the RTX 2070 precision path even on newer test GPUs.
        with patch.object(runner, "precision", return_value=torch.float16):
            (self.directory / "data").mkdir()
            data = runner.dataset("relative_offset_copy", self.cfg, 7, self.directory / "data")
            worker = runner.Worker("relative_offset_copy", "rope", 7, self.cfg,
                                   self.directory / "gpu", data, "cuda:0")
            worker.scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 24)
            before = {k: v.clone() for k, v in worker.model.state_dict().items()}
            worker.train_step()
            self.assertEqual(worker.state["step"], 1)
            self.assertLess(worker.scaler.get_scale(), 2.0 ** 24)
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in worker.model.state_dict().items()))
            worker.save()
            resumed = runner.Worker("relative_offset_copy", "rope", 7, self.cfg,
                                    self.directory / "gpu", data, "cuda:0")
            self.assertEqual(worker.scaler.state_dict(), resumed.scaler.state_dict())
            worker.train_step()
            resumed = runner.Worker("relative_offset_copy", "rope", 7, self.cfg,
                                    self.directory / "gpu", data, "cuda:0")
            resumed.train_step()
            for key, value in worker.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)

    def setUp(self):
        self.cfg = runner.read_json(runner.HERE / "experiment.json")
        self.cfg.update(seeds=[7], encodings=["nope", "rope"], batch_size=4,
                        micro_batch_size=2, eval_batch_size=2, steps_per_epoch=1,
                        eval_every_epochs=1, patience=2, val_samples=4, test_samples=4,
                        cl_stage_epochs=2, cl_stages=2, lab_test_lengths=[16, 40],
                        kv_test_lengths=[2, 5], sorting_test_lengths=[2, 5])
        self.cfg["model"].update(n_layer=1, n_head=2, n_embd=16)
        base = runner.HERE / "results" / "verification"
        base.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(dir=base))

    def worker(self, name="relative_offset_copy", enc="nope", directory=None):
        directory = directory or self.directory
        directory.mkdir(parents=True, exist_ok=True)
        data = runner.dataset(name, self.cfg, 7, directory)
        return runner.Worker(name, enc, 7, self.cfg, directory, data, "cpu")

    @staticmethod
    def score(acc):
        return dict(acc=acc, correct=int(acc * 4), n=4, loss=1.0)

    def test_inclusive_curriculum_and_lengths(self):
        cfg = runner.read_json(runner.HERE / "experiment.json")
        for name, key in [("kv_retrieval", "n_pairs"), ("sorting", "n_items")]:
            self.assertEqual(runner.task_params(name, cfg, stage=1)[key], (2, 3))
            self.assertEqual(runner.task_params(name, cfg, stage=25)[key], (2, 51))
        runner.validate(cfg, ["kv_retrieval", "sorting", "relative_offset_copy"])

    def test_all_encodings_backward_and_ood(self):
        task = get_task("relative_offset_copy", {"seed": 7})
        for enc in runner.read_json(runner.HERE / "experiment.json")["encodings"]:
            with self.subTest(encoding=enc):
                model = Model(Config(**self.cfg["model"], vocab_size=task.vocab_size, pos_encoding=enc))
                x, y = runner.batch(task, task.sample(2), "cpu")
                _, loss = model(x, y)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                long_task = get_task("relative_offset_copy", {"length_range": (256, 256)})
                result = runner.evaluate(model, long_task, long_task.sample(2), "cpu", 1)
                self.assertEqual(result["n"], 2)

    def test_resume_matches_uninterrupted_and_alignment(self):
        first = self.worker(directory=self.directory / "a")
        with patch.object(runner, "evaluate", return_value=self.score(0.5)):
            first.advance()
            resumed = self.worker(directory=self.directory / "a")
            resumed.advance()
            expected = self.worker(directory=self.directory / "b")
            expected.advance()
            expected.advance()
            for key, value in resumed.model.state_dict().items():
                self.assertTrue(torch.equal(value, expected.model.state_dict()[key]), key)
            resumed.advance()
            self.assertEqual(resumed.state["stop_epoch"], 3)
            self.assertEqual(resumed.state["stop_reason"], "patience")
            resumed.advance(target=5)
            resumed = self.worker(directory=self.directory / "a")
            resumed.advance(target=5)
            self.assertEqual(resumed.state["epoch"], 5)
            self.assertTrue(resumed.state["aligned"])
            self.assertEqual(resumed.state["stop_epoch"], 3)

    def test_cl_ignores_patience_and_first_stage_target(self):
        worker = self.worker("kv_retrieval")
        with patch.object(runner, "evaluate", return_value=self.score(1.0)):
            worker.advance()
            self.assertFalse(worker.state["discovered"])
            worker.advance()
            self.assertEqual(worker.state["stage"], 2)
        with patch.object(runner, "evaluate", return_value=self.score(0.0)):
            for _ in range(3):
                worker.advance()
            self.assertFalse(worker.state["discovered"])
        with patch.object(runner, "evaluate", return_value=self.score(1.0)):
            worker.advance()
        self.assertEqual(worker.state["stop_epoch"], 6)
        self.assertEqual(worker.state["stop_reason"], "val_acc_1")

    def test_dataset_disjoint_and_test_resume(self):
        worker = self.worker()
        sets = [set(map(runner.digest, items)) for items in [*worker.data["val"].values(), *worker.data["test"].values()]]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        worker.advance(target=1)
        worker.test()
        rows = runner.read_json(worker.directory / "acc_L.json")
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["split"] for r in rows}, {"ID", "OOD"})
        self.assertTrue(all(r["n"] == 4 for r in rows))
        with patch.object(runner, "evaluate", side_effect=AssertionError("test point repeated")):
            worker.test()

    def test_task_scheduler_equalizes_epochs(self):
        # Different discovery times; every final checkpoint must use the maximum.
        name = "relative_offset_copy"
        directory = self.directory / name
        directory.mkdir()
        first = self.worker(name, "nope", directory)
        second = self.worker(name, "rope", directory)
        with patch.object(runner, "evaluate", return_value=self.score(1.0)):
            first.advance()
        with patch.object(runner, "evaluate", return_value=self.score(0.0)):
            for _ in range(3):
                second.advance()
            for _ in runner.run_task(name, self.cfg, self.directory, "cpu"):
                pass
        for enc in self.cfg["encodings"]:
            state = runner.status(directory, 7, enc)
            self.assertEqual(state["epoch"], 3)
            self.assertTrue(state["aligned"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
