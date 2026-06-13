import datetime
import json
from typing import Optional, List, Dict, Any
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class TrialResult(Base):
    __tablename__ = "trial_results"

    trial_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    epoch_reached: Mapped[int] = mapped_column(Integer, nullable=False)
    primary_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    primary_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score_history_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    weights_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    gpu_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    max_vram_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    oom_triggered: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    failure_tag: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # e.g. "NAN_LOSS", "OOM", etc.
    worker_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    git_commit: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    dataset_version: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    health_tier: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    health_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    def get_history(self) -> List[Dict[str, Any]]:
        if not self.score_history_json:
            return []
        try:
            return json.loads(self.score_history_json)
        except Exception:
            return []

    def set_history(self, history: List[Dict[str, Any]]):
        self.score_history_json = json.dumps(history)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "study_name": self.study_name,
            "epoch_reached": self.epoch_reached,
            "primary_score": self.primary_score,
            "primary_loss": self.primary_loss,
            "score_history": self.get_history(),
            "weights_path": self.weights_path,
            "gpu_model": self.gpu_model,
            "max_vram_gb": self.max_vram_gb,
            "oom_triggered": self.oom_triggered,
            "failure_tag": self.failure_tag,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "worker_id": self.worker_id,
            "git_commit": self.git_commit,
            "dataset_version": self.dataset_version,
            "health_tier": self.health_tier,
            "health_reason": self.health_reason,
        }

class TrialMetadata(Base):
    __tablename__ = "trial_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[int] = mapped_column(Integer, nullable=False)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    meta_key: Mapped[str] = mapped_column(String(100), nullable=False)
    meta_value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trial_id": self.trial_id,
            "study_name": self.study_name,
            "meta_key": self.meta_key,
            "meta_value": self.meta_value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class SystemConfiguration(Base):
    __tablename__ = "system_configuration"

    study_name: Mapped[str] = mapped_column(String(200), primary_key=True)
    config_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    config_value: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "config_key": self.config_key,
            "config_value": self.config_value,
            "version": self.version,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class CompactedPacket(Base):
    __tablename__ = "compacted_packets"
    __table_args__ = (UniqueConstraint('study_name', 'trials_evaluated', name='_study_trials_uc'),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    trials_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    packet_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

class StudyCard(Base):
    __tablename__ = "study_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    card_type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "recap", "model_card", "synthesis"
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "study_name": self.study_name,
            "card_type": self.card_type,
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class AgentReasoningLog(Base):
    __tablename__ = "agent_reasoning_logs"

    trial_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_outcome_rationale: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_score_improvement: Mapped[float] = mapped_column(Float, nullable=False)
    actual_score_improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "study_name": self.study_name,
            "model_version": self.model_version,
            "prompt_strategy": self.prompt_strategy,
            "predicted_outcome_rationale": self.predicted_outcome_rationale,
            "estimated_score_improvement": self.estimated_score_improvement,
            "actual_score_improvement": self.actual_score_improvement,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class StudyReview(Base):
    __tablename__ = "study_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    health_rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    policy_action: Mapped[str] = mapped_column(String(60), nullable=False, default="no_change")
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, default="unspecified")
    prompt_strategy: Mapped[str] = mapped_column(String(100), nullable=False, default="coordinator_review")
    reasons_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of trigger reasons
    trials_evaluated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # idempotency window key
    estimated_score_improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cited_best_trial: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="high")
    baseline_best_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    applied_at_completed_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    applied_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    actual_score_improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outcome_measured_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    outcome_status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    quality_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    def get_reasons(self) -> List[Dict[str, Any]]:
        if not self.reasons_json:
            return []
        try:
            return json.loads(self.reasons_json)
        except Exception:
            return []

    def set_reasons(self, reasons: Optional[List[Dict[str, Any]]]):
        self.reasons_json = json.dumps(reasons or [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "study_name": self.study_name,
            "health_rating": self.health_rating,
            "summary": self.summary,
            "policy_action": self.policy_action,
            "model_version": self.model_version,
            "prompt_strategy": self.prompt_strategy,
            "reasons": self.get_reasons(),
            "trials_evaluated": self.trials_evaluated,
            "estimated_score_improvement": self.estimated_score_improvement,
            "cited_best_trial": self.cited_best_trial,
            "confidence": self.confidence or "high",
            "baseline_best_score": self.baseline_best_score,
            "applied_at_completed_count": self.applied_at_completed_count,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "actual_score_improvement": self.actual_score_improvement,
            "outcome_measured_at": self.outcome_measured_at.isoformat() if self.outcome_measured_at else None,
            "outcome_status": self.outcome_status,
            "quality_flagged": self.quality_flagged,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class StudyStatus(Base):
    __tablename__ = "study_status"

    study_name: Mapped[str] = mapped_column(String(200), primary_key=True)
    health_tier: Mapped[str] = mapped_column(String(50), default="healthy")
    health_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    health_updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )
    nudge_dismissed_trials: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "health_tier": self.health_tier,
            "health_reason": self.health_reason,
            "health_updated_at": self.health_updated_at.isoformat() if self.health_updated_at else None,
            "nudge_dismissed_trials": self.nudge_dismissed_trials,
        }

class InvalidProposal(Base):
    __tablename__ = "invalid_proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    invalid_parameters: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string of parameters proposed
    validation_error: Mapped[str] = mapped_column(Text, nullable=False)  # Reason/exception string
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    def get_parameters(self) -> Dict[str, Any]:
        try:
            return json.loads(self.invalid_parameters)
        except Exception:
            return {}

    def set_parameters(self, params: Dict[str, Any]):
        self.invalid_parameters = json.dumps(params)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "study_name": self.study_name,
            "model_version": self.model_version,
            "prompt_strategy": self.prompt_strategy,
            "invalid_parameters": self.get_parameters(),
            "validation_error": self.validation_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class TrialLease(Base):
    __tablename__ = "trial_leases"

    trial_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    leased_to: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    lease_expires_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "study_name": self.study_name,
            "leased_to": self.leased_to,
            "lease_expires_at": self.lease_expires_at.isoformat() if self.lease_expires_at else None
        }

class CoordinatorMetric(Base):
    __tablename__ = "coordinator_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    action_taken: Mapped[str] = mapped_column(String(100), nullable=False)
    trials_at_review: Mapped[int] = mapped_column(Integer, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "study_name": self.study_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "action_taken": self.action_taken,
            "trials_at_review": self.trials_at_review
        }

class SuggestMetric(Base):
    __tablename__ = "suggest_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)  # new_trial, reclaimed_lease, recycled_running

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "study_name": self.study_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latency_ms": self.latency_ms,
            "source": self.source
        }
