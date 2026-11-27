"""PresenceDetectionData — ConsumerData subclass for presence detection results."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vt_radar.data.consumer_data import ConsumerData


@dataclass
class PresenceDetectionData(ConsumerData):
    """Data produced by PresenceDetectionConsumer.

    Holds results from both MotionEnergyPresence and DBSCANPresence detectors,
    plus a fused (OR / max-confidence) aggregate.
    """

    consumer_name: str = field(default="PresenceDetection")
    motion_present: bool = False
    motion_confidence: float = 0.0
    dbscan_present: bool = False
    dbscan_confidence: float = 0.0
    fused_present: bool = False
    fused_confidence: float = 0.0
    debug: Optional[Dict[str, Any]] = None

    def serialize(self) -> Dict[str, Any]:
        """Serialize for JSON transmission to the frontend."""
        return {
            "motion_present": self.motion_present,
            "motion_confidence": self.motion_confidence,
            "dbscan_present": self.dbscan_present,
            "dbscan_confidence": self.dbscan_confidence,
            "fused_present": self.fused_present,
            "fused_confidence": self.fused_confidence,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "PresenceDetectionData":
        """Deserialize from a dictionary."""
        return cls(
            motion_present=data.get("motion_present", False),
            motion_confidence=data.get("motion_confidence", 0.0),
            dbscan_present=data.get("dbscan_present", False),
            dbscan_confidence=data.get("dbscan_confidence", 0.0),
            fused_present=data.get("fused_present", False),
            fused_confidence=data.get("fused_confidence", 0.0),
        )
