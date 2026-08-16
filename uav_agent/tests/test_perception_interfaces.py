"""Pure-Python tests for perception interfaces and honest placeholders."""

from __future__ import annotations

import inspect
import unittest

from perception import (
    DetectorTrackerPerception,
    OraclePerception,
    PerceptionBackend,
    VLMVerifier,
)


class PerceptionInterfaceTests(unittest.TestCase):
    def test_oracle_perception_structurally_satisfies_backend(self) -> None:
        oracle = OraclePerception(target_id="target_0")

        self.assertIsInstance(oracle, PerceptionBackend)
        self.assertTrue(callable(oracle.observe))
        self.assertTrue(callable(oracle.get_observation))
        self.assertEqual(
            inspect.signature(oracle.observe).parameters["frame"].annotation,
            "object",
        )

    def test_detector_tracker_is_structural_backend_but_not_implemented(self) -> None:
        backend = DetectorTrackerPerception()

        self.assertIsInstance(backend, PerceptionBackend)
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            backend.observe(object())

    def test_vlm_verifier_is_explicitly_not_implemented(self) -> None:
        verifier = VLMVerifier()

        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            verifier.verify(object())


if __name__ == "__main__":
    unittest.main()
