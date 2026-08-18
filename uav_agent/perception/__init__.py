"""Visual perception, runtime-policy, and target-confirmation interfaces."""

from perception.base import (
    DetectorTrackerBackend,
    IdentityVerifierBackend,
    PerceptionBackend,
    SemanticVerifierBackend,
)
from perception.candidate_bank import (
    CandidateBank,
    CandidateLifecycle,
    CandidateReviewRef,
    CandidateSnapshot,
)
from perception.confirmation import (
    CandidateConfirmationCoordinator,
    CandidateConfirmationError,
    ConfirmationDecision,
    ConfirmationPolicy,
    ConfirmationResult,
)
from perception.detector_tracker import DetectorTrackerPerception
from perception.grounding import (
    CandidateResolutionUnavailable,
    CandidateResolver,
    GroundingBackend,
    GroundingBackendUnavailable,
    GroundingProposal,
    LearnedGrounder,
    OracleEvaluationCandidateResolver,
    OracleEvaluationGrounder,
    OraclePositionProvider,
    ProductionCandidateResolver,
    QwenVLGrounder,
    ResolvedCandidatePosition,
    YOLOEGrounder,
)
from perception.oracle import OraclePerception, OraclePerceptionError
from perception.prompt_types import (
    DeterministicPromptCompiler,
    PromptBundle,
    TargetPromptAdapter,
)
from perception.qwen_vlm_verifier import (
    QwenVLMVerifier,
    VisualReviewFrame,
    VisualReviewInput,
)
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
from perception.visual_review import (
    QwenVisualReview,
    ReviewDisposition,
    VisualReviewAcceptance,
    VisualReviewAction,
    VisualReviewCandidate,
    VisualReviewDecision,
    VisualReviewExpectation,
    VisualReviewGate,
    VisualReviewMode,
    VisualReviewProtocolError,
    build_qwen_visual_review_json_schema,
)

__all__ = [
    "CandidateBank",
    "CandidateLifecycle",
    "CandidateReviewRef",
    "CandidateSnapshot",
    "CandidateConfirmationCoordinator",
    "CandidateConfirmationError",
    "ConfirmationDecision",
    "ConfirmationPolicy",
    "ConfirmationResult",
    "DetectionCandidate",
    "DetectorTrackerBackend",
    "DetectorTrackerPerception",
    "DeterministicPromptCompiler",
    "GuardedPerceptionBackend",
    "GroundingBackend",
    "GroundingBackendUnavailable",
    "GroundingProposal",
    "IdentityConsistencyEvidence",
    "IdentityVerifierBackend",
    "OraclePerception",
    "OraclePerceptionError",
    "OracleEvaluationCandidateResolver",
    "OracleEvaluationGrounder",
    "OraclePositionProvider",
    "PerceptionBoundaryError",
    "PerceptionBackend",
    "PerceptionCapability",
    "PerceptionRuntimeProfile",
    "PromptBundle",
    "ProductionCandidateResolver",
    "QwenVLMVerifier",
    "QwenVLGrounder",
    "ReIDVerifier",
    "QwenVisualReview",
    "ReviewDisposition",
    "SemanticVerification",
    "SemanticVerifierBackend",
    "ShortTrackEvidence",
    "TargetPromptAdapter",
    "CandidateResolutionUnavailable",
    "CandidateResolver",
    "LearnedGrounder",
    "ResolvedCandidatePosition",
    "VLMVerifier",
    "VisualReviewAcceptance",
    "VisualReviewAction",
    "VisualReviewCandidate",
    "VisualReviewDecision",
    "VisualReviewExpectation",
    "VisualReviewFrame",
    "VisualReviewGate",
    "VisualReviewInput",
    "VisualReviewMode",
    "VisualReviewProtocolError",
    "YOLOEGrounder",
    "build_qwen_visual_review_json_schema",
    "observation_contains_oracle_data",
    "validate_observation_access",
]
