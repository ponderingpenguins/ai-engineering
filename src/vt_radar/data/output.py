from __future__ import annotations

from loguru import logger

import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .raw_radar_data import RawRadarData

if TYPE_CHECKING:
    import pandas as pd
    from .rangedoppler import RangeDoppler


class TrackData(np.ndarray):
    def __new__(cls, trackdata):
        obj = np.asarray(trackdata).view(cls)
        return obj

    def to_pandas(self) -> pd.DataFrame:
        import pandas as pd
        if len(self):
            return pd.DataFrame(self, columns=["target_id", "x", "y", "z", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "G", "conf", "None1", "None2", "None3", "None4"])
        return pd.DataFrame()
    
    def to_list(self):
        return self.flatten().tolist(), self.shape

    @staticmethod
    def from_list(data_tuple):
        data, shape = data_tuple
        # Ensure shape is a tuple of integers
        shape = tuple(map(int, shape))
        return TrackData(np.array(data).reshape(shape))
        
class PointCloud(np.ndarray):
    """A class representing a 3D point cloud with additional metadata.
    
    This class extends numpy's ndarray to provide specialized functionality for radar point cloud data.
    Each point in the cloud contains 7 attributes: x, y, z coordinates, Doppler velocity, SNR, Noise, and Track index.
    
    Attributes:
        _pointmeta: Optional metadata associated with each point in the cloud
    """
    
    def __new__(cls, pointcloud: np.ndarray, pointmeta: Optional[List[Dict[str, Any]]] = None) -> 'PointCloud':
        """Create a new PointCloud instance.
        
        Args:
            pointcloud: A numpy array containing the point cloud data
            pointmeta: Optional list of dictionaries containing metadata for each point
            
        Returns:
            A new PointCloud instance
        """
        obj = np.asarray(pointcloud).view(cls)
        return obj
    
    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        """Called when a new array is created from an existing one.

        Args:
            obj: The original array being finalized
        """
        if obj is None: return
        # numpy calls this (not __init__) for every array derived from a
        # PointCloud — slice, .copy(), .view(), arithmetic, np.concatenate, etc.
        # Without propagating _pointmeta the derived array lacks the attribute
        # entirely, so to_pandas() (which reads self._pointmeta) raises
        # AttributeError. Inherit the source's metadata, defaulting to None.
        self._pointmeta = getattr(obj, "_pointmeta", None)

    def __init__(self, pointcloud: np.ndarray, pointmeta: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize the PointCloud instance.
        
        Args:
            pointcloud: A numpy array containing the point cloud data
            pointmeta: Optional list of dictionaries containing metadata for each point
        """
        self._pointmeta = pointmeta

    def plot_3d(self) -> None:
        """Plot the point cloud in 3D using matplotlib.

        Creates a 3D scatter plot of the points with predefined axis limits and labels.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        nice = fig.add_subplot(111, projection='3d')

        nice.set_zlim3d(bottom=-5, top=5)
        nice.set_ylim(bottom=0, top=10)
        nice.set_xlim(left=-4, right=4)
        nice.set_xlabel('X Label')
        nice.set_ylabel('Y Label')
        nice.set_zlabel('Z Label')

        # PointCloud is (N, 7); x/y/z are COLUMNS 0/1/2. self[0]/self[1]/self[2]
        # are the first three points' full rows, so the old call scattered three
        # points' attribute vectors against each other. Use the columns (matching
        # get_xy_coordinates' self[:, :2]).
        nice.scatter(self[:, 0], self[:, 1], self[:, 2], c='r', marker='o', s=2)
        plt.show()

    def to_list(self) -> Tuple[List[float], Tuple[int, ...]]:
        """Convert the point cloud to a flattened list and its original shape.
        
        Returns:
            A tuple containing:
            - The flattened point cloud data as a list
            - The original shape of the point cloud
        """
        return self.flatten().tolist(), self.shape
    
    def get_xy_coordinates(self) -> np.ndarray:
        """Extract X and Y coordinates from the point cloud.
        
        Returns:
            A numpy array containing only the X and Y coordinates (first two columns)
        """
        return self[:, :2]

    @staticmethod
    def from_list(data_tuple: Tuple[List[float], Tuple[int, ...]]) -> 'PointCloud':
        """Create a PointCloud instance from a flattened list and shape.
        
        Args:
            data_tuple: A tuple containing:
                - The flattened point cloud data as a list
                - The desired shape of the point cloud
                
        Returns:
            A new PointCloud instance with the specified shape
        """
        data, shape = data_tuple
        shape = tuple(map(int, shape))
        return PointCloud(np.array(data).reshape(shape))

    def to_pandas(self) -> pd.DataFrame:
        """Convert the point cloud to a pandas DataFrame.

        Returns:
            A pandas DataFrame containing the point cloud data and metadata
        """
        import pandas as pd
        df1 = pd.DataFrame(self, columns=["x", "y", "z", "Doppler", "SNR", "Noise", "Track index"])
        # _pointmeta defaults to None (and is often empty); from_records(None)
        # raises TypeError, so return just the point columns when there's no meta.
        if not self._pointmeta:
            return df1
        df2 = pd.DataFrame.from_records(self._pointmeta)
        return pd.concat([df1, df2], axis=1)
    
class TargetIndexes(np.ndarray):
    def __new__(cls, trackindexes):
        obj = np.asarray(trackindexes).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return

    def to_list(self):
        return self.flatten().tolist(), self.shape

    @staticmethod
    def from_list(data_tuple):
        data, shape = data_tuple
        # Ensure shape is a tuple of integers
        shape = tuple(map(int, shape))
        return TargetIndexes(np.array(data).reshape(shape))

    def to_pandas(self):
        import pandas as pd
        df1 = pd.DataFrame(self, columns=["target_id"])
        return df1

                            
class ProcessedRadarData(object):
    def __init__(self, raw_data=None, output_dict=None, pointcloud=None, trackdata=None):
        """Initialize ProcessedRadarData with either RawRadarData, dictionary, or individual components.
        
        Args:
            raw_data: RawRadarData object (preferred interface)
            output_dict: Dictionary containing output data (legacy interface)
            pointcloud: Point cloud data as numpy array
            trackdata: Track data as numpy array
        """
        if raw_data is not None:
            # New interface - process RawRadarData
            self._error = raw_data.error
            self._frame_num = raw_data.frame_num
            self._height_data = raw_data.height_data if raw_data.height_data is not None else []
            self._pointcloud = raw_data.point_cloud if raw_data.point_cloud is not None else []
            self._num_detected_heights = raw_data.num_detected_heights
            self._num_detected_points = raw_data.num_detected_points
            self._num_detected_tracks = raw_data.num_detected_tracks
            self._pointmeta = raw_data.point_meta if raw_data.point_meta is not None else []
            self._presence_indication = raw_data.presence_indication
            self._trackdata = raw_data.track_data if raw_data.track_data is not None else []
            self._track_indexes = raw_data.track_indexes if raw_data.track_indexes is not None else []
            self._range_doppler_heatmap = raw_data.range_doppler_heatmap if raw_data.range_doppler_heatmap is not None else []
            self._range_profile = raw_data.range_profile if raw_data.range_profile is not None else []
            self._stats = raw_data.stats if raw_data.stats is not None else []
            self._temperature_stats = raw_data.temperature_stats
            self._vitals = raw_data.vitals
            # Store any remaining data for potential future use
            self._remaining_data = raw_data.remaining_data
        elif output_dict is not None:
            # Legacy interface - process dictionary
            self._output = output_dict
            self._error = self._output.pop("error", None)
            self._frame_num = self._output.pop("frameNum", "x")
            self._height_data = self._output.pop("heightData", [])
            self._pointcloud = self._output.pop("pointCloud", [])
            self._num_detected_heights = self._output.pop("numDetectedHeights", 0)
            self._num_detected_points = self._output.pop("numDetectedPoints", 0)
            self._num_detected_tracks = self._output.pop("numDetectedTracks", 0)
            self._pointmeta = self._output.pop("pointMeta", [])
            self._presence_indication = self._output.pop("presenceIndication", None)
            self._trackdata = self._output.pop("trackData", [])
            self._track_indexes = self._output.pop("trackIndexes", [])
            self._range_doppler_heatmap = self._output.pop("rangeDopplerHeatmap", [])
            self._range_profile = self._output.pop("rangeProfile", [])
            self._stats = self._output.pop("stats", [])
            self._temperature_stats = self._output.pop("temperatureStats", None)        
            self._vitals = self._output.pop("vitals", None)

            rest = self._output.keys()
            if len(rest):
                logger.error("Keys left in output dict {}", rest)
        else:
            # New interface - direct assignment
            self._pointcloud = pointcloud
            self._trackdata = trackdata
            self._error = None
            self._frame_num = "x"
            self._height_data = []
            self._num_detected_heights = 0
            self._num_detected_points = 0
            self._num_detected_tracks = 0
            self._pointmeta = []
            self._presence_indication = None
            self._track_indexes = []
            self._range_doppler_heatmap = []
            self._range_profile = []
            self._stats = []
            self._temperature_stats = None        
            self._vitals = None

    def __str__(self):
        return "Frame #" + str(self._frame_num)

    def as_dict(self):
        retval =  {
            "error" : self._error,
            "frameNum" : self._frame_num,
            "heightData" : self._height_data,
            "pointCloud" : self._pointcloud,
            "numDetectedPoints" : self._num_detected_points,
            "numDetectedTracks" : self._num_detected_tracks,
            "trackData" : self._trackdata,
            "trackIndexes" : self._track_indexes
        }
        if self._vitals:
            retval["vitals"] = self._vitals

        return retval

    def to_json_dict(self):
        """Convert to dictionary with JSON-serializable values (numpy arrays → lists)."""
        retval = {
            "error": self._error,
            "frameNum": self._frame_num,
            "numDetectedPoints": self._num_detected_points,
            "numDetectedTracks": self._num_detected_tracks,
        }
        
        # Convert numpy arrays to lists
        if self._height_data is not None:
            retval["heightData"] = self._height_data.tolist() if hasattr(self._height_data, 'tolist') else self._height_data
        else:
            retval["heightData"] = []
            
        if self._pointcloud is not None:
            retval["pointCloud"] = self._pointcloud.tolist() if hasattr(self._pointcloud, 'tolist') else self._pointcloud
        else:
            retval["pointCloud"] = []
            
        if self._trackdata is not None:
            retval["trackData"] = self._trackdata.tolist() if hasattr(self._trackdata, 'tolist') else self._trackdata
        else:
            retval["trackData"] = []
            
        if self._track_indexes is not None:
            retval["trackIndexes"] = self._track_indexes.tolist() if hasattr(self._track_indexes, 'tolist') else self._track_indexes
        else:
            retval["trackIndexes"] = []
        
        # Handle vitals with nested array conversion
        if self._vitals:
            vitals_dict = self._vitals.copy() if isinstance(self._vitals, dict) else {}
            if 'heartWaveform' in vitals_dict and vitals_dict['heartWaveform'] is not None:
                vitals_dict['heartWaveform'] = vitals_dict['heartWaveform'].tolist() if hasattr(vitals_dict['heartWaveform'], 'tolist') else vitals_dict['heartWaveform']
            if 'breathWaveform' in vitals_dict and vitals_dict['breathWaveform'] is not None:
                vitals_dict['breathWaveform'] = vitals_dict['breathWaveform'].tolist() if hasattr(vitals_dict['breathWaveform'], 'tolist') else vitals_dict['breathWaveform']
            retval["vitals"] = vitals_dict
        else:
            retval["vitals"] = None

        return retval

    @property
    def error(self):
        return self._error

    @property
    def frame_num(self):
        return self._frame_num

    @property
    def heightdata(self):
        return self._height_data
    
    @property
    def num_detected_heights(self):
        return self._num_detected_heights

    @property
    def num_detected_points(self):
        return self._num_detected_points
    
    @property
    def num_detected_tracks(self):
        return self._num_detected_tracks

    @property
    def pointcloud(self) -> Optional[PointCloud]:
        if self._pointcloud is None:
            return None
        return PointCloud(self._pointcloud, self._pointmeta)
    
    @pointcloud.setter
    def pointcloud(self, new_pointcloud):
        self._pointcloud = new_pointcloud

    @property
    def range_doppler_heatmap(self) -> RangeDoppler:
        return self._range_doppler_heatmap
    
    @range_doppler_heatmap.setter
    def range_doppler_heatmap(self, new_range_doppler_heatmap):
        self._range_doppler_heatmap = new_range_doppler_heatmap

    @property
    def trackdata(self) -> Optional[TrackData]:
        if self._trackdata is None:
            return None
        return TrackData(self._trackdata)
    
    @trackdata.setter
    def trackdata(self, new_trackdata):
        self._trackdata = new_trackdata
    
    @property
    def trackindexes(self) -> TargetIndexes:
        return TargetIndexes(self._track_indexes)
    
    @trackindexes.setter
    def trackindexes(self, new_trackindexes):
        self._track_indexes = new_trackindexes
