"""FallDetectionData — ConsumerData subclass for fall detection results."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vt_radar.data.consumer_data import ConsumerData


@dataclass
class FallDetectionData(ConsumerData):
    """Data produced by FallDetectionConsumer."""

    consumer_name: str = field(default="FallDetection")
    detected: bool = False
    confidence: float = 0.0
    algorithm: Optional[str] = None
    track_id: Optional[int] = None

    def serialize(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "confidence": self.confidence,
            "algorithm": self.algorithm,
            "track_id": self.track_id,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "FallDetectionData":
        return cls(
            detected=data.get("detected", False),
            confidence=data.get("confidence", 0.0),
            algorithm=data.get("algorithm"),
            track_id=data.get("track_id"),
        )
