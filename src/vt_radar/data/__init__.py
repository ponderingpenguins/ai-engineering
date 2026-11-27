"""Data module for radar data structures and processing.

This module provides comprehensive data structures and utilities for handling
radar data in the Valtes Radar Toolbox. It includes classes for frames,
datasets, point clouds, and various radar data representations.

Main Components:
    - Frame: Core data structure for radar frames
    - Dataset: Classes for reading and managing radar datasets
    - PointCloud: 3D point cloud representation and processing
    - Image: Image data handling and processing
    - Output: Processed radar output data structures
    - RadarFrame: Raw radar data processing and beamforming
    - RangeCube: Range processing results
    - RangeDoppler: Range-Doppler processing results

Example:
    Basic usage with radar frames:

    .. code-block:: python

        from vt_radar.data import Frame, TLVDataset
        from vt_radar.config.ti_config import ConfigFile

        # Load configuration
        config = ConfigFile()
        config.parseCfg('path/to/config.cfg')

        # Create dataset
        dataset = TLVDataset('path/to/dataset')

        # Process frames
        for frame in dataset:
            point_cloud = frame.output.pointcloud
            track_data = frame.output.trackdata
            print(f"Frame {frame.timestamp}: {len(point_cloud)} points")
"""

# Lazy submodule loading (PEP 562). Importing vt_radar.data — or any of
# its submodules — must NOT eager-import the heavy science/vision stack (cv2, pandas,
# matplotlib, sklearn, h5py). Public names resolve on first access via __getattr__,
# so a lean deploy (e.g. the streaming Pi) can physically omit the submodules it never
# touches. See docs/superpowers/specs/2026-07-03-streaming-lean-pi-design.md.
import importlib
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:  # give type-checkers the real names without importing at runtime
    from .frame import Frame
    from .image import Image
    from .output import ProcessedRadarData
    from .raw_radar_data import RawRadarData
    from .pointcloud import PointCloud
    from .pointcloud_window import PointCloudWindow
    from .clustering import Cluster, Clusterer, ClusteringResult, DBSCANClusterer
    from .dataset import TLVDataset, BaseDataset
    from .consumer_data import ConsumerData
    from .labels import Labels, LabelTrack, LabelSchema

_LAZY_ATTRS = {
    "Frame": ".frame",
    "Image": ".image",
    "ProcessedRadarData": ".output",
    "RawRadarData": ".raw_radar_data",
    "PointCloud": ".pointcloud",
    "PointCloudWindow": ".pointcloud_window",
    "Cluster": ".clustering",
    "Clusterer": ".clustering",
    "ClusteringResult": ".clustering",
    "DBSCANClusterer": ".clustering",
    "TLVDataset": ".dataset",
    "BaseDataset": ".dataset",
    "ConsumerData": ".consumer_data",
    "Labels": ".labels",
    "LabelTrack": ".labels",
    "LabelSchema": ".labels",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    'Frame', 'Image', 'ProcessedRadarData', 'RawRadarData', 'PointCloud',
    'PointCloudWindow',
    'Cluster', 'Clusterer', 'ClusteringResult', 'DBSCANClusterer',
    'TLVDataset', 'BaseDataset', 'ConsumerData',
    'Labels', 'LabelTrack', 'LabelSchema',
]
