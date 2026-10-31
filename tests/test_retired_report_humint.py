"""Named vertical doors are absent; generic audit/governance remain."""
from relata.audit import AuditClient, AsyncAuditClient
from relata.governance import GovernanceClient, AsyncGovernanceClient

def test_report_named_methods_are_absent() -> None:
    for cls in [AuditClient,AsyncAuditClient]:
        assert not hasattr(cls,"sign_receipt")
        assert not hasattr(cls,"export_pdf")
        assert not hasattr(cls,"entries")
        assert not hasattr(cls,"find_by_request_id")
        assert hasattr(cls,"log")

def test_humint_named_methods_are_absent() -> None:
    for cls in [GovernanceClient,AsyncGovernanceClient]:
        for name in ["request_breakglass","approve_breakglass","breakglass_status"]:
            assert not hasattr(cls,name)
        assert hasattr(cls,"place_legal_hold")
