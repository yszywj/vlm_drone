"""Visual perception, runtime-policy, and target-confirmation interfaces."""

from perception.base import (
    DetectorTrackerBackend,
    IdentityVerifierBackend,
    PerceptionBackend,
    SemanticVerifierBackend,
)
from perception.confirmation import (
    CandidateConfirmationCoordinator,
    CandidateConfirmationError,
    ConfirmationDecision,
    ConfirmationPolicy,
    ConfirmationResult,
)
from perception.detector_tracker import DetectorTrackerPerception
from perception.oracle import OraclePerception, OraclePerceptionError
from perception.reid_verifier import ReIDVerifier
from perception.runtime import (
    GuardedPerceptionBackend,
    PerceptionBoundaryError,
    PerceptionCapability,
    PerceptionRuntimeProfile,
    observation_contains_oracle_data,
    validate_observation_access,
)
from perception.types import (
    DetectionCandidate,
    IdentityConsistencyEvidence,
    SemanticVerification,
    ShortTrackEvidence,
)
from perception.vlm_verifier import VLMVerifier

__all__ = [
    "CandidateConfirmationCoordinator",
    "CandidateConfirmationError",
    "ConfirmationDecision",
    "ConfirmationPolicy",
    "ConfirmationResult",
    "DetectionCandidate",
    "DetectorTrackerBackend",
    "DetectorTrackerPerception",
    "GuardedPerceptionBackend",
    "IdentityConsistencyEvidence",
    "IdentityVerifierBackend",
    "OraclePerception",
    "OraclePerceptionError",
    "PerceptionBoundaryError",
    "PerceptionBackend",
    "PerceptionCapability",
    "PerceptionRuntimeProfile",
    "ReIDVerifier",
    "SemanticVerification",
    "SemanticVerifierBackend",
    "ShortTrackEvidence",
    "VLMVerifier",
    "observation_contains_oracle_data",
    "validate_observation_access",
]
