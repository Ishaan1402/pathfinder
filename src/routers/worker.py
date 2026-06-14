from fastapi import APIRouter
from ..suggest import SuggestRequest, handle_api_suggest_trial
from ..leases import HeartbeatRequest, handle_api_heartbeat
from ..reporting import ReportEpochRequest, CompleteTrialRequest, handle_api_report_epoch, handle_api_complete_trial

router = APIRouter(prefix="/api")

@router.get("/suggest_trial")
def api_suggest_trial_help():
    """Browser/curl GET lands here — suggest requires POST (Colab worker uses POST)."""
    return {
        "error": "Method not allowed: use POST, not GET",
        "post_url": "/api/suggest_trial",
        "body_example": {
            "study_name": "bridge_crack_study",
            "reasoning": "Autonomous worker suggestion request.",
        },
        "curl_example": (
            'curl -X POST "$BROKER_URL/api/suggest_trial" '
            '-H "Content-Type: application/json" '
            '-d \'{"study_name":"bridge_crack_study"}\''
        ),
    }

@router.post("/suggest_trial")
@router.post("/suggest_trials")  # common typo alias
def api_suggest_trial(req: SuggestRequest):
    return handle_api_suggest_trial(req)

@router.post("/heartbeat")
def api_heartbeat(req: HeartbeatRequest):
    return handle_api_heartbeat(req)

@router.post("/report_epoch")
def api_report_epoch(req: ReportEpochRequest):
    return handle_api_report_epoch(req)

@router.post("/complete_trial")
def api_complete_trial(req: CompleteTrialRequest):
    return handle_api_complete_trial(req)
