"""Future visual-language candidate verification boundary.

The verifier will eventually confirm candidate targets, match requested
attributes, and disambiguate candidates during reacquisition.  This Stage-0
placeholder deliberately loads no model and never returns a fabricated
verification result.
"""

from __future__ import annotations

from perception.types import DetectionCandidate, SemanticVerification


class VLMVerifier:
    """Placeholder for candidate confirmation and reacquisition disambiguation."""

    def verify(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Reject every call until a real verification contract exists.

        No argument schema is frozen yet because candidate-result types are
        deliberately outside Task 4.  Accepting arbitrary arguments here
        ensures every attempted use fails with the promised explicit error,
        rather than an incidental signature ``TypeError``.
        """

        del args, kwargs
        raise NotImplementedError(
            "VLMVerifier is not implemented; no visual candidate verification "
            "result is available"
        )

    def verify_candidate(
        self,
        candidate: DetectionCandidate,
        target_description: str,
        camera_rgb: object,
    ) -> SemanticVerification:
        del candidate, target_description, camera_rgb
        raise NotImplementedError(
            "VLMVerifier is not implemented; no semantic verification result "
            "is available"
        )


__all__ = ["VLMVerifier"]
