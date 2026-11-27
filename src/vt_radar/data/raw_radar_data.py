"""Raw radar data structures for TLV parsing.

This module contains the RawRadarData class that represents the raw data
structure returned by parseStandardFrame.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class RawRadarData:
    """Raw radar data structure containing parsed TLV data.
    
    This class represents the raw data structure returned by parseStandardFrame,
    containing all the parsed TLV data in a structured format.
    
    Attributes:
        frame_num: Frame number from radar
        error: Error code (0 = no error)
        point_cloud: Point cloud data as numpy array (N, 7) - X, Y, Z, Doppler, SNR, Noise, TrackIndex
        num_detected_points: Number of detected points
        track_data: Track data as numpy array (N, 16) - target_id, pos, vel, acc, etc.
        num_detected_tracks: Number of detected tracks
        track_indexes: Track indexes for point cloud association
        height_data: Height data as numpy array
        num_detected_heights: Number of detected heights
        range_profile: Range profile data
        range_doppler_heatmap: Range-Doppler heatmap data
        stats: Performance statistics
        temperature_stats: Temperature statistics
        vitals: Vital signs data
        presence_indication: Presence indication data
        occupancy: Occupancy state machine data
        enhanced_presence_det: Enhanced presence detection data
        classifier_output: Classifier output data
        gesture: Gesture data
        gesture_neural_net_prob: Gesture neural network probabilities
        features: Gesture features
        point_meta: Point metadata for compressed points
        range_profile_major: Major range profile
        range_profile_minor: Minor range profile
        remaining_data: Any remaining unparsed data
    """
    
    # Core frame data
    frame_num: int
    error: int = 0
    
    # Point cloud data
    point_cloud: Optional[np.ndarray] = None
    num_detected_points: int = 0
    
    # Track data
    track_data: Optional[np.ndarray] = None
    num_detected_tracks: int = 0
    track_indexes: Optional[np.ndarray] = None
    
    # Height data
    height_data: Optional[np.ndarray] = None
    num_detected_heights: int = 0
    
    # Profile and heatmap data
    range_profile: Optional[List[float]] = None
    range_doppler_heatmap: Optional[np.ndarray] = None
    range_profile_major: Optional[List[float]] = None
    range_profile_minor: Optional[List[float]] = None
    
    # Statistics and monitoring
    stats: Optional[Tuple[int, ...]] = None
    temperature_stats: Optional[Tuple[int, ...]] = None
    
    # Vital signs and health monitoring
    vitals: Optional[Dict[str, Any]] = None
    
    # Presence and occupancy detection
    presence_indication: Optional[Tuple[int, ...]] = None
    occupancy: Optional[List[bool]] = None
    enhanced_presence_det: Optional[List[int]] = None
    
    # Classification and gesture recognition
    classifier_output: Optional[np.ndarray] = None
    gesture: Optional[int] = None
    gesture_neural_net_prob: Optional[Tuple[float, ...]] = None
    features: Optional[List[float]] = None
    
    # Metadata
    point_meta: Optional[List[Dict[str, Any]]] = None
    
    # Remaining unparsed data
    remaining_data: Optional[bytes] = None
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any], remaining_data: Optional[bytes] = None) -> 'RawRadarData':
        """Create RawRadarData from the dictionary returned by parseStandardFrame.
        
        Args:
            data_dict: Dictionary from parseStandardFrame
            remaining_data: Any remaining unparsed data
            
        Returns:
            RawRadarData: Structured raw radar data
        """
        return cls(
            frame_num=data_dict.get('frameNum', 0),
            error=data_dict.get('error', 0),
            point_cloud=data_dict.get('pointCloud'),
            num_detected_points=data_dict.get('numDetectedPoints', 0),
            track_data=data_dict.get('trackData'),
            num_detected_tracks=data_dict.get('numDetectedTracks', 0),
            track_indexes=data_dict.get('trackIndexes'),
            height_data=data_dict.get('heightData'),
            num_detected_heights=data_dict.get('numDetectedHeights', 0),
            range_profile=data_dict.get('rangeProfile'),
            range_doppler_heatmap=data_dict.get('rangeDopplerHeatmap'),
            range_profile_major=data_dict.get('rangeProfileMajor'),
            range_profile_minor=data_dict.get('rangeProfileMinor'),
            stats=data_dict.get('stats'),
            temperature_stats=data_dict.get('temperatureStats'),
            vitals=data_dict.get('vitals'),
            presence_indication=data_dict.get('presenceIndication'),
            occupancy=data_dict.get('occupancy'),
            enhanced_presence_det=data_dict.get('enhancedPresenceDet'),
            classifier_output=data_dict.get('classifierOutput'),
            gesture=data_dict.get('gesture'),
            gesture_neural_net_prob=data_dict.get('gestureNeuralNetProb'),
            features=data_dict.get('features'),
            point_meta=data_dict.get('pointMeta'),
            remaining_data=remaining_data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RawRadarData to dictionary format for backward compatibility.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the raw radar data
        """
        return {
            'frameNum': self.frame_num,
            'error': self.error,
            'pointCloud': self.point_cloud,
            'numDetectedPoints': self.num_detected_points,
            'trackData': self.track_data,
            'numDetectedTracks': self.num_detected_tracks,
            'trackIndexes': self.track_indexes,
            'heightData': self.height_data,
            'numDetectedHeights': self.num_detected_heights,
            'rangeProfile': self.range_profile,
            'rangeDopplerHeatmap': self.range_doppler_heatmap,
            'rangeProfileMajor': self.range_profile_major,
            'rangeProfileMinor': self.range_profile_minor,
            'stats': self.stats,
            'temperatureStats': self.temperature_stats,
            'vitals': self.vitals,
            'presenceIndication': self.presence_indication,
            'occupancy': self.occupancy,
            'enhancedPresenceDet': self.enhanced_presence_det,
            'classifierOutput': self.classifier_output,
            'gesture': self.gesture,
            'gestureNeuralNetProb': self.gesture_neural_net_prob,
            'features': self.features,
            'pointMeta': self.point_meta
        }