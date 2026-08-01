from __future__ import annotations

from tools.audit_ape_runtime import _language_branch_evidence


def test_language_branch_evidence_does_not_claim_strict_completeness() -> None:
    evidence = _language_branch_evidence(
        {
            "largest_prefixes": [
                {
                    "prefix": "model_vision.model_language.net",
                    "tensor_count": 390,
                }
            ]
        }
    )

    assert evidence["branch_present"] is True
    assert evidence["tensor_count"] == 390
    assert evidence["strict_load_verified"] is False


def test_language_branch_evidence_handles_missing_prefix() -> None:
    evidence = _language_branch_evidence({"largest_prefixes": []})

    assert evidence["branch_present"] is False
    assert evidence["tensor_count"] == 0
    assert evidence["strict_load_verified"] is False
