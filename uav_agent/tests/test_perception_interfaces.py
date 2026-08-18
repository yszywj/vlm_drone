"""Pure-Python tests for perception interfaces and honest placeholders."""

from __future__ import annotations

import inspect
import unittest

from perception import (
    DetectorTrackerBackend,
    DetectorTrackerPerception,
    IdentityVerifierBackend,
    OraclePerception,
    PerceptionBackend,
    ReIDVerifier,
    SemanticVerifierBackend,
    VLMVerifier,
)


class PerceptionInterfaceTests(unittest.TestCase):
    def test_oracle_perception_structurally_satisfies_backend(self) -> None:
        oracle = OraclePerception(uav_id="uav_1", target_id="target_0")

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
        self.assertIsInstance(backend, DetectorTrackerBackend)
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            backend.observe(object())
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            backend.detect(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            backend.update_track("candidate", object())  # type: ignore[arg-type]

    def test_vlm_verifier_is_explicitly_not_implemented(self) -> None:
        verifier = VLMVerifier()

        self.assertIsInstance(verifier, SemanticVerifierBackend)
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            verifier.verify(object())
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            verifier.verify_candidate(object(), "target", object())  # type: ignore[arg-type]

    def test_reid_verifier_is_explicitly_not_implemented(self) -> None:
        verifier = ReIDVerifier()

        self.assertIsInstance(verifier, IdentityVerifierBackend)
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            verifier.verify_identity("candidate", object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
