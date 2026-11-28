import unittest

from schema_study.formal_methods import (
    assistant_labels,
    checkpoint_can_resume,
    combine_head_losses,
    curriculum_stages,
    distributed_accumulation,
    execution_authorised,
    kl_consistency,
    manifest_payload,
    nested_document_subsets,
    node_hours,
    parent_event_probability,
    principal_training_counts,
    schema_view,
    training_requirement,
)


class FormalMethodTests(unittest.TestCase):
    def test_accumulation_keeps_the_global_batch(self):
        self.assertEqual(distributed_accumulation(1, 1), 16)
        self.assertEqual(distributed_accumulation(2, 2), 4)
        with self.assertRaises(ValueError):
            distributed_accumulation(3, 1)

    def test_only_the_assistant_span_is_supervised(self):
        labels = assistant_labels([1, 2, 3, 0, 0], prompt_tokens=2, pad_positions={4})
        self.assertEqual(labels, [-100, -100, 3, 0, -100])
        with self.assertRaises(ValueError):
            assistant_labels([0, 0], prompt_tokens=2, pad_positions=set())

    def test_losses_use_frozen_definitions(self):
        self.assertEqual(combine_head_losses({"existence": 1.0, "coarse": 2.0, "native": 3.0}), 6.0)
        self.assertAlmostEqual(kl_consistency([0.0, 1.0], [0.5, 0.5]), 0.693147, places=5)
        self.assertAlmostEqual(parent_event_probability([0.5, 0.5]), 0.75)
        self.assertNotEqual(parent_event_probability([0.5, 0.5]), 1.0)

    def test_schema_curriculum_and_nested_subsets(self):
        views = ["E0/P0", "E2/P3"]
        self.assertEqual(schema_view("candidate", 1, views), schema_view("candidate", 1, views))
        self.assertEqual(curriculum_stages("coarse_to_fine")[-1], "native")
        self.assertEqual(curriculum_stages("fine_to_coarse")[-1], "native")
        subsets = nested_document_subsets([f"d{i}" for i in range(100)])
        self.assertTrue(set(subsets["10"]).issubset(subsets["25"]))
        self.assertTrue(set(subsets["25"]).issubset(subsets["50"]))
        self.assertEqual(len(subsets["100"]), 100)

    def test_reuse_and_resume_do_not_restart_a_valid_run(self):
        self.assertEqual(training_requirement("drugprot", "native", 0, True), "reuse_historical_checkpoint")
        self.assertEqual(training_requirement("biored", "E2", 710, True), "retrain")
        saved = {
            "model_state": {"weight": 1}, "optimizer_state": {"step": 10}, "scheduler_state": {}, "scaler_state": {},
            "rng_state": {}, "step": 10, "epoch": 1, "data_fingerprint": "data", "configuration_fingerprint": "config",
        }
        self.assertTrue(checkpoint_can_resume(saved, "config", "data"))
        self.assertFalse(checkpoint_can_resume(saved, "other", "data"))

    def test_inventory_blocks_submission_and_keeps_the_matrix_separate(self):
        counts = principal_training_counts(True, 12)
        self.assertEqual(counts["designed_supervised_runs"], 324)
        self.assertEqual(counts["new_principal_training_runs"], 312)
        self.assertEqual(counts["bc8_training_runs"], 0)
        self.assertEqual(counts["drugprot_new_encoder_runs"], 0)
        payload = manifest_payload()
        self.assertEqual(payload["gpu_submission"], "blocked")
        self.assertEqual(payload["nli_baseline"]["id2label"]["0"], "entailment")
        self.assertFalse(payload["biolinkbert"]["weights_local"])
        self.assertAlmostEqual(node_hours(4), 1.0)
        self.assertFalse(execution_authorised("/nonexistent/study"))


if __name__ == "__main__":
    unittest.main()
