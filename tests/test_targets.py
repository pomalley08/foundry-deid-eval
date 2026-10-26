import tempfile
import time
import unittest
from pathlib import Path

from deid_eval import dataset, targets


class FakeModelTarget(targets.Target):
    def __init__(self, name="fake-model", fail_on=None):
        self.name = name
        self.model = name
        self.calls = 0
        self.fail_on = fail_on

    def deidentify(self, text):
        self.calls += 1
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("simulated failure")
        return {"deid_text": targets.baseline_deidentify(text), "input_tokens": 100, "output_tokens": 90}


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes = dataset.generate(seed=42, size=6)

    def test_result_shape(self):
        result = targets.run_target(targets.BaselineTarget(), self.notes[:1])[0]
        for key in ("deid_text", "latency_ms", "input_tokens", "output_tokens", "target", "model"):
            self.assertIn(key, result)

    def test_baseline_runs_offline_with_null_tokens(self):
        results = targets.run_target(targets.BaselineTarget(), self.notes)
        self.assertEqual(len(results), len(self.notes))
        for note, result in zip(self.notes, results):
            self.assertTrue(result["deid_text"])
            self.assertNotEqual(result["deid_text"], note["source_text"])
            self.assertIsNone(result["input_tokens"])
            self.assertIsNone(result["output_tokens"])
            self.assertIsNone(result["error"])

    def test_one_failure_does_not_stop_the_run(self):
        fake = FakeModelTarget(fail_on=self.notes[2]["source_text"][:40])
        results = targets.run_target(fake, self.notes)
        self.assertEqual(fake.calls, len(self.notes))
        self.assertIn("simulated failure", results[2]["error"])
        self.assertEqual(results[2]["deid_text"], "")
        self.assertTrue(all(r["deid_text"] for i, r in enumerate(results) if i != 2))

    def test_model_targets_skipped_without_endpoint(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "", "AZURE_OPENAI_DEPLOYMENTS": ""}):
            built, message = targets.model_targets()
        self.assertEqual(built, [])
        self.assertIn("skipped", message)

    def test_cache_mode_makes_no_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "data" / "cache"
            writer = FakeModelTarget()
            targets.collect_results(self.notes, [writer], use_cache=False, cache_dir=cache_dir, log=lambda *_: None)
            self.assertTrue(targets.cache_path("fake-model", cache_dir).exists())

            replay = FakeModelTarget()
            uncached = FakeModelTarget(name="other-model")
            results = targets.collect_results(
                self.notes, [replay, uncached], use_cache=True, cache_dir=cache_dir, log=lambda *_: None
            )
            self.assertEqual(replay.calls, 0)
            self.assertEqual(uncached.calls, 0)
            self.assertIn("fake-model", results)
            self.assertNotIn("other-model", results)

    def test_stale_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "data" / "cache"
            targets.collect_results(self.notes, [FakeModelTarget()], use_cache=False, cache_dir=cache_dir, log=lambda *_: None)
            other_notes = dataset.generate(seed=7, size=6)
            results = targets.collect_results(other_notes, [], use_cache=True, cache_dir=cache_dir, log=lambda *_: None)
            self.assertEqual(results, {})

    def test_model_target_uses_shared_prompt_and_temperature_zero(self):
        calls = []

        class Usage:
            prompt_tokens, completion_tokens = 120, 80

        class Response:
            model = "gpt-test-2025"
            usage = Usage()
            choices = [type("C", (), {"message": type("M", (), {"content": "```\n[PATIENT_NAME] seen\n```"})()})()]

        class Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        calls.append(kwargs)
                        return Response()

        target = targets.ModelTarget(Client(), "my-deployment")
        output = target.deidentify("Ana Ruiz seen")
        self.assertEqual(calls[0]["temperature"], 0)
        self.assertEqual(calls[0]["model"], "my-deployment")
        self.assertEqual(calls[0]["messages"][0]["content"], targets.DEID_PROMPT)
        self.assertEqual(output["deid_text"], "[PATIENT_NAME] seen")
        self.assertEqual((output["input_tokens"], output["output_tokens"]), (120, 80))
        result = targets.run_target(target, self.notes[:1])[0]
        self.assertEqual(result["prompt_sha"], targets.PROMPT_SHA)

    def test_prompt_lineage(self):
        baseline = targets.run_target(targets.BaselineTarget(), self.notes[:1])[0]
        self.assertIsNone(baseline["prompt_sha"])
        self.assertEqual(targets.PROMPT_SHA, targets.prompt_hash(targets.DEID_PROMPT))
        self.assertNotEqual(targets.prompt_hash(targets.DEID_PROMPT + " "), targets.PROMPT_SHA)

        class Failing(targets.ModelTarget):
            def deidentify(self, text):
                raise RuntimeError("down")

        failed = targets.run_target(Failing(None, "m", concurrency=1), self.notes[:1])[0]
        self.assertEqual(failed["prompt_sha"], targets.PROMPT_SHA)
        self.assertIn("down", failed["error"])

    def test_temperature_fallback_under_concurrency(self):
        class Response:
            model = "reasoning-model"
            usage = None

            def __init__(self, text):
                self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]

        class Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        if "temperature" in kwargs:
                            time.sleep(0.05)
                            raise ValueError("Unsupported value: 'temperature' does not support 0")
                        return Response(kwargs["messages"][1]["content"])

        target = targets.ModelTarget(Client(), "reasoning", concurrency=4)
        results = targets.run_target(target, self.notes)
        self.assertTrue(all(r["error"] is None for r in results), [r["error"] for r in results])
        self.assertTrue(all(r["temperature"] is None for r in results))


if __name__ == "__main__":
    unittest.main()
