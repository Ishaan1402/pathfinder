import pytest
from pydantic import ValidationError
from src.reporting import ReportEpochRequest, CompleteTrialRequest

def test_report_epoch_single_metric():
    """Verify that ReportEpochRequest accepts single metrics and rejects empty metrics."""
    # Only score
    req1 = ReportEpochRequest(study_name="test", trial_id=1, epoch=1, score=0.95)
    assert req1.score == 0.95
    assert req1.loss is None
    
    # Only loss
    req2 = ReportEpochRequest(study_name="test", trial_id=1, epoch=1, loss=0.05)
    assert req2.score is None
    assert req2.loss == 0.05
    
    # Both
    req3 = ReportEpochRequest(study_name="test", trial_id=1, epoch=1, score=0.9, loss=0.1)
    assert req3.score == 0.9
    assert req3.loss == 0.1
    
    # Neither should fail
    with pytest.raises(ValidationError):
        ReportEpochRequest(study_name="test", trial_id=1, epoch=1)

def test_complete_trial_single_metric():
    """Verify that CompleteTrialRequest accepts single metrics and rejects empty metrics."""
    history = [{"epoch": 1, "score": 0.95}]
    
    # Only score
    req1 = CompleteTrialRequest(study_name="test", trial_id=1, epoch=1, score=0.95, weights_path="", history=history)
    assert req1.score == 0.95
    assert req1.loss is None
    
    # Neither should fail
    with pytest.raises(ValidationError):
        CompleteTrialRequest(study_name="test", trial_id=1, epoch=1, weights_path="", history=history)
