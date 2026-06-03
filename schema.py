import datetime
import json
from typing import Optional, List, Dict, Any
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    version_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    crack_surface_type: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "asphalt", "concrete"
    resolution_width: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution_height: Mapped[int] = mapped_column(Integer, nullable=False)
    total_images: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "crack_surface_type": self.crack_surface_type,
            "resolution_width": self.resolution_width,
            "resolution_height": self.resolution_height,
            "total_images": self.total_images,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class SegmentationMetric(Base):
    __tablename__ = "segmentation_metrics"

    trial_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    epoch_reached: Mapped[int] = mapped_column(Integer, nullable=False)
    final_bce_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_dice_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    val_loss_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    weights_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    def get_history(self) -> List[Dict[str, Any]]:
        if not self.val_loss_history:
            return []
        try:
            return json.loads(self.val_loss_history)
        except Exception:
            return []

    def set_history(self, history: List[Dict[str, Any]]):
        self.val_loss_history = json.dumps(history)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "epoch_reached": self.epoch_reached,
            "final_bce_loss": self.final_bce_loss,
            "final_dice_score": self.final_dice_score,
            "val_loss_history": self.get_history(),
            "weights_path": self.weights_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class AgentReasoningLog(Base):
    __tablename__ = "agent_reasoning_logs"

    trial_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_outcome_rationale: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_dice_improvement: Mapped[float] = mapped_column(Float, nullable=False)
    actual_dice_improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
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
            "estimated_dice_improvement": self.estimated_dice_improvement,
            "actual_dice_improvement": self.actual_dice_improvement,
            "created_at": self.created_at.isoformat() if self.created_at else None,
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
