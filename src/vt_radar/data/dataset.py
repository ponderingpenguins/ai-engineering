"""Dataset module for radar data management and processing.

This module provides comprehensive dataset management capabilities for radar data,
including reading from HDF5 files, filtering by timestamp, and various dataset
types for different radar data formats.

Main Components:
    - BaseDataset: Base class for dataset operations
    - ValtesDataset: Main dataset class for TLV data
    - TLVDataset: Specialized dataset for TLV format
    - RawDataset: Dataset for raw ADC data
    - RawDataset2: Alternative raw dataset implementation

Utility Functions:
    - read_hdf5: Read frames from HDF5 files
    - read_directory: Read frames from directory with filtering
    - apply_timestamp_filter: Filter files by timestamp range
    - get_directory_files: Get files from directory with sampling
    - load_radar_config: Load radar configuration from file
"""
import glob
from datetime import datetime
import random
import pandas as pd
import functools
import glob
import importlib
import os
import numpy as np
import h5py
import platform
from loguru import logger
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from vt_radar.data.labels import Labels
from vt_radar.data.frame import Frame
from vt_radar.config.ti_config import ConfigFile
from itertools import takewhile

import matplotlib.pyplot as plt
from vt_radar.mmwave.dataloader import DCA1000


def _try_import_dataset_index() -> Optional[Any]:
    """Import ``vt_radar.tools.dataset_index`` if it is available.

    ``vt_radar.tools`` is a toolbox-only convenience (dataset name -> path
    registry/index) and is not part of the IP-clean recorder subset. Name-based
    dataset resolution is best-effort: when the toolbox isn't installed, callers
    fall back to their existing "not found" / empty-result handling instead of
    crashing with ImportError. Uses ``importlib`` (rather than a static
    ``from vt_radar.tools... import ...``) so this optional dependency isn't
    mistaken for a hard one.
    """
    try:
        return importlib.import_module("vt_radar.tools.dataset_index")
    except ImportError:
        return None


def apply_timestamp_filter(filename: str, min_timestamp: Optional[datetime] = None,
                          max_timestamp: Optional[datetime] = None) -> bool:
    """Filter files based on timestamp range.

    This function efficiently filters HDF5 files based on their timestamp range
    without reading the entire file. It checks if the file's timestamp range
    overlaps with the specified min/max timestamps.

    Args:
        filename: Path to the HDF5 file to check
        min_timestamp: Minimum timestamp for filtering (inclusive)
        max_timestamp: Maximum timestamp for filtering (inclusive)

    Returns:
        bool: True if the file should be included based on timestamp filtering,
            False otherwise

    Note:
        This function reads only the first and last timestamps from the file
        to minimize I/O overhead when filtering large datasets.

    Example:
        .. code-block:: python

            # Filter files from last 24 hours
            from datetime import datetime, timedelta

            min_time = datetime.now() - timedelta(hours=24)
            files = [f for f in all_files if apply_timestamp_filter(f, min_time)]
    """
    if min_timestamp is None and max_timestamp is None:
        return True

    # Convert np.datetime64 and datetime to pd.Timestamp
    # (see https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64)
    file_min_timestamp = pd.Timestamp(pd.read_hdf(filename, stop=1)['timestamp'].values[0])
    file_max_timestamp = pd.Timestamp(pd.read_hdf(filename, start=-1)['timestamp'].values[0])
    if min_timestamp is not None:
        min_timestamp = pd.Timestamp(min_timestamp)
    if max_timestamp is not None:
        max_timestamp = pd.Timestamp(max_timestamp)

    if min_timestamp is not None and max_timestamp is not None:
        return (min_timestamp <= file_min_timestamp <= max_timestamp) or \
               (min_timestamp <= file_max_timestamp <= max_timestamp) or \
               (file_min_timestamp <= min_timestamp <= file_max_timestamp) or \
               (file_min_timestamp <= max_timestamp <= file_max_timestamp)
    elif min_timestamp is not None:
        return (min_timestamp <= file_min_timestamp) or (min_timestamp <= file_max_timestamp)
    elif max_timestamp is not None:
        return (file_min_timestamp <= max_timestamp) or (file_max_timestamp <= max_timestamp)



def read_hdf5(filename: str, config: Optional[ConfigFile] = None) -> Iterator[Frame]:
    """Read frames from an HDF5 file.

    Args:
        filename: Path to the HDF5 file
        config: Optional radar configuration for frame processing

    Yields:
        Frame: Individual radar frames from the file

    Raises:
        Exception: If the file cannot be read or processed

    Example:
        .. code-block:: python

            # Read all frames from a file
            for frame in read_hdf5('data.h5', config):
                print(f"Frame: {frame.timestamp}")
    """
    try:
        df = pd.read_hdf(filename)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        logger.exception(f"Could not read {filename}")
        return

    for index, row in df.iterrows():
        # Convert pandas Series to dictionary format expected by Frame.from_raw
        row_dict = {
            'timestamp': row['timestamp'],
            'frame': row['frame'],
            'image': row.get('image', None),
            'mmwave': row.get('mmwave', None)
        }
        frame = Frame.from_raw(row_dict, config)
        yield frame

def read_directory(dirname: str, config: Optional[ConfigFile] = None, 
                  read_func: Any = read_hdf5, min_timestamp: Optional[datetime] = None, 
                  max_timestamp: Optional[datetime] = None, sample_size: Optional[int] = None) -> Iterator[Frame]:
    """Read frames from a directory with optional filtering and sampling.

    Args:
        dirname: Directory path containing data files
        config: Optional radar configuration for frame processing
        read_func: Function to use for reading individual files
        min_timestamp: Minimum timestamp for filtering
        max_timestamp: Maximum timestamp for filtering
        sample_size: Optional number of files to randomly sample

    Yields:
        Frame: Individual radar frames from the directory

    Example:
        .. code-block:: python

            # Read all frames from directory
            for frame in read_directory('data/', config):
                print(f"Frame: {frame.timestamp}")

            # Read frames from last 24 hours
            from datetime import datetime, timedelta
            min_time = datetime.now() - timedelta(hours=24)
            for frame in read_directory('data/', config, min_timestamp=min_time):
                print(f"Frame: {frame.timestamp}")
    """
    files = get_directory_files(dirname, sample_size=sample_size)

    timestamp_passed = False
    
    def stop_condition(filename: str) -> bool:
        nonlocal timestamp_passed

        result = apply_timestamp_filter(filename=filename, min_timestamp=min_timestamp, max_timestamp=max_timestamp)
        if result:
            timestamp_passed = True
        return result or not timestamp_passed

    for filename in takewhile(stop_condition, files):
        if apply_timestamp_filter(filename=filename, min_timestamp=min_timestamp, max_timestamp=max_timestamp):
            frame: Frame
            # Use the injected reader (defaults to read_hdf5). Hardcoding
            # read_hdf5 here silently ignored a caller-supplied read_func.
            for frame in read_func(filename, config):
                yield frame

def get_directory_files(dirname: str, sample_size: Optional[int] = None, extension: str = "*.h5") -> List[str]:
    """Get files from a directory with optional random sampling.

    Args:
        dirname: Directory path to search
        sample_size: Optional number of files to randomly sample
        extension: File extension pattern to match

    Returns:
        List[str]: Sorted list of file paths

    Example:
        .. code-block:: python

            # Get all HDF5 files
            files = get_directory_files('data/')

            # Get 10 random files
            files = get_directory_files('data/', sample_size=10)
    """
    files = sorted(glob.glob(os.path.join(dirname, extension), recursive=False))

    if sample_size is not None:
        # Clamp to what's available: random.sample() raises ValueError if sample_size
        # exceeds the number of files (e.g. a small dataset), so cap it and return all.
        files = sorted(random.sample(files, min(sample_size, len(files))))

    return files

def _get_repo_root() -> str:
    """Find the repository root by walking up from this file."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return d


def _get_google_drive_dir(_type: str) -> str:
    """Get the Google Drive base directory path for datasets."""
    if platform.system() == "Windows":
        return os.path.join("G:", os.sep, "Shared drives", "Data", _type)
    elif platform.system() == "Linux":
        return os.path.join("/mnt/g/Shared drives/Data/", _type)
    else:
        raise OSError(f"Unsupported platform: {platform.system()}")


def get_all_base_dirs(_type: str = "TLV") -> List[str]:
    """Get all candidate base directories for datasets, in priority order.

    Priority:
        1. RADARTOOLBOX_DATA_DIR env var (if set and exists)
        2. {repo_root}/datasets/{type} (local convention)
        3. Google Drive path (platform-specific)

    Args:
        _type: Dataset type (e.g., "TLV", "DCA1000")

    Returns:
        List[str]: Candidate directories in priority order (may not all exist)
    """
    dirs: List[str] = []

    env_dir = os.environ.get("RADARTOOLBOX_DATA_DIR")
    if env_dir:
        dirs.append(os.path.join(env_dir, _type))

    repo_root = _get_repo_root()
    dirs.append(os.path.join(repo_root, "datasets", _type))

    dirs.append(_get_google_drive_dir(_type))

    return dirs


def get_base_dir(_type: str = "TLV") -> str:
    """Get the base directory path for datasets based on platform.

    Returns the first existing directory from the priority list
    (env var > local datasets/ > Google Drive), falling back to the
    Google Drive path if nothing exists locally.

    Args:
        _type: Dataset type (e.g., "TLV", "DCA1000")

    Returns:
        str: Base directory path
    """
    for d in get_all_base_dirs(_type):
        if os.path.isdir(d):
            return d

    # Fallback: Google Drive path even if it doesn't exist (preserves old behavior)
    return _get_google_drive_dir(_type)

def load_radar_config(config_file: str) -> Optional[ConfigFile]:
    """Load radar configuration from file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Optional[ConfigFile]: Loaded configuration object, or None if loading fails

    Example:
        .. code-block:: python

            config = load_radar_config('radar_config.cfg')
            if config is not None:
                print("Configuration loaded successfully")
    """
    try:
        cfg = ConfigFile()
        cfg.parseCfg(config_file)
        return cfg
    except Exception as e:
        if os.path.isdir(config_file):
            logger.error(f"Config file '{config_file}' is a directory")
        logger.warning(f"Failed to load config from '{config_file}': {e}")
        return None

class BaseDataset(object):
    """Base class for radar dataset operations.

    This class provides common functionality for all dataset types,
    including file discovery and basic dataset operations.

    Attributes:
        _basedir (str): Base directory path for the dataset type

    Example:
        .. code-block:: python

            # Create base dataset
            dataset = BaseDataset("TLV")
            files = dataset.get_directory_files("path/to/data")
    """

    def __init__(self, dataset_path: str, config: Optional[ConfigFile] = None, config_file: Optional[str] = None) -> None:
        """Initialize the base dataset.

        Args:
            dataset_path: Path to the dataset directory
            config: Optional radar configuration
            config_file: Optional path to configuration file
        """
        self.dataset_path = dataset_path
        if config is not None:
            self.config = config
        elif config_file is not None:
            self.config = load_radar_config(config_file)
        else:
            # Look for .cfg files in the dataset directory
            cfg_files = glob.glob(os.path.join(dataset_path, "*.cfg"))
            if cfg_files:
                self.config = load_radar_config(cfg_files[0])
            else:
                self.config = None

    def __iter__(self):
        """Abstract method for iteration - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement __iter__")

    def __len__(self):
        """Abstract method for length - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement __len__")

    def get_directory_files(self, dirname: str, extension: str = "*.h5") -> List[str]:
        """Get files from a directory with error checking.

        Args:
            dirname: Directory path to search
            extension: File extension pattern to match

        Returns:
            List[str]: Sorted list of file paths

        Note:
            This method logs an error if no files are found in the directory.
        """
        files = sorted(glob.glob(os.path.join(dirname, extension), recursive=False))
        if len(files) == 0:
            logger.error("Empty dataset {}, no {} data files", dirname, extension)
            logger.error(os.listdir(dirname))
        return files

class ValtesDataset(BaseDataset):
    """Main dataset class for TLV radar data.

    This class provides comprehensive dataset management for TLV radar data,
    including file discovery, configuration loading, and frame iteration with
    optional filtering and sampling.

    Attributes:
        _dataset_name (str): Name of the dataset directory
        _dir (str): Full path to the dataset directory
        _debug (bool): Debug mode flag
        _config (Optional[ConfigFile]): Loaded radar configuration
        _data_files (List[str]): List of data files in the dataset
        _cfg_files (List[str]): List of configuration files
        _min_timestamp (Optional[datetime]): Minimum timestamp for filtering
        _max_timestamp (Optional[datetime]): Maximum timestamp for filtering
        _sample_size (Optional[int]): Number of files to randomly sample

    Example:
        .. code-block:: python

            # Create dataset
            dataset = ValtesDataset(
                data_dir="my_dataset",
                type_="TLV",
                min_timestamp=datetime.now() - timedelta(hours=1)
            )

            # Iterate through frames
            for frame in dataset:
                print(f"Frame: {frame.timestamp}")

            # Get random frame
            random_frame = dataset.random_frame()
    """

    def __init__(self, data_dir: str, debug: bool = False, type_: str = "TLV", 
                 min_timestamp: Optional[datetime] = None, max_timestamp: Optional[datetime] = None, 
                 sample_size: Optional[int] = None) -> None:
        """Initialize the Valtes dataset.

        Args:
            data_dir: Name of the dataset directory
            debug: Enable debug mode for additional logging
            type_: Dataset type (e.g., "TLV", "DCA1000")
            min_timestamp: Minimum timestamp for filtering frames
            max_timestamp: Maximum timestamp for filtering frames
            sample_size: Number of files to randomly sample

        Raises:
            AssertionError: If no configuration files are found
        """
        self._basedir = get_base_dir(type_)
        self._dataset_name = data_dir
        # Resolve path: absolute dirs pass through, otherwise search all base dirs
        if os.path.isabs(data_dir) and os.path.isdir(data_dir):
            self._dir = data_dir
        else:
            resolved = None
            dataset_index = _try_import_dataset_index()
            if dataset_index is not None:
                for base in get_all_base_dirs(type_):
                    resolved = dataset_index.resolve_recording_path(data_dir, base)
                    if resolved:
                        break
            self._dir = resolved or os.path.join(self._basedir, data_dir)
        super().__init__(self._dir)
        self._debug = debug
        # Always assign the filter/sample state so a missing directory yields a
        # consistent *empty* dataset instead of a half-initialized object whose
        # read_frames()/frames/__len__/random_frame() all AttributeError.
        self._min_timestamp = min_timestamp
        self._max_timestamp = max_timestamp
        self._sample_size = sample_size
        if not os.path.exists(self._dir):
            logger.error("Not a directory: '{}'", self._dir)
            self._cfg_files = []
            self._data_files = []
            self._config = None
            return
        self._read_configs()
        assert(self._cfg_files)
        self._config = self.load_config() or None
        self._read_data()
        self._min_timestamp = min_timestamp
        self._max_timestamp = max_timestamp
        self._sample_size = sample_size

    def _read_data(self) -> None:
        """Read data files from the dataset directory."""
        self._data_files = self.get_directory_files(self._dir, "*.h5")

    def _read_configs(self) -> None:
        """Read configuration files from the dataset directory."""
        self._cfg_files = self.get_directory_files(self._dir, "*.cfg")

    def __iter__(self) -> Iterator[Frame]:
        """Iterate through frames in the dataset.

        Yields:
            Frame: Individual radar frames from the dataset

        Note:
            Frames are filtered by timestamp and sampling if specified
            during initialization.
        """
        frame: Frame
        for frame in read_directory(self._dir, config=self.config, 
                                   min_timestamp=self._min_timestamp, 
                                   max_timestamp=self._max_timestamp, 
                                   sample_size=self._sample_size):
            yield frame

    def __len__(self) -> int:
        """Get the number of frames in the dataset.

        Returns:
            int: Number of frames in the dataset
        """
        return len(self.frames)

    def load_config(self) -> Optional[ConfigFile]:
        """Load radar configuration from the first available config file.

        Returns:
            Optional[ConfigFile]: Loaded configuration, or None if no config files found
        """
        if len(self._cfg_files):
            return load_radar_config(self._cfg_files[0])
        logger.warning("No config file available for dataset {}", self)
        return None

    @property
    def config(self) -> Optional[ConfigFile]:
        """Get the loaded radar configuration.

        Returns:
            Optional[ConfigFile]: Radar configuration, or None if not loaded
        """
        return self._config
    
    @config.setter
    def config(self, value: Optional[ConfigFile]) -> None:
        """Set the radar configuration.

        Args:
            value: Radar configuration to set
        """
        self._config = value
    
    def read_frames(self) -> List[Frame]:
        """Read all frames from the dataset into memory.

        Returns:
            List[Frame]: List of all frames in the dataset

        Note:
            This method loads all frames into memory, which may be
            memory-intensive for large datasets. Use iteration for
            memory-efficient processing.
        """
        frames = list(read_directory(self._dir, config=self.config, 
                                    min_timestamp=self._min_timestamp, 
                                    max_timestamp=self._max_timestamp, 
                                    sample_size=self._sample_size))
        return frames

    @functools.cached_property
    def frames(self) -> List[Frame]:
        """Get all frames in the dataset (cached property).

        Returns:
            List[Frame]: List of all frames in the dataset

        Note:
            This property caches the result after first access.
            Use read_frames() for explicit control over caching.
        """
        return self.read_frames()

    def random_frame(self) -> Frame:
        """Get a random frame from the dataset.

        Returns:
            Frame: Randomly selected frame from the dataset

        Raises:
            IndexError: If the dataset is empty
        """
        # random.randint's upper bound is inclusive, so randint(0, len) could
        # return len -> IndexError. random.choice picks a valid index and
        # raises IndexError on an empty dataset (matching the docstring).
        return random.choice(self.frames)


class TLVDataset(BaseDataset):
    """Specialized dataset class for TLV radar data.

    This class is a convenience wrapper around ValtesDataset specifically
    for TLV format radar data.

    Example:
        .. code-block:: python

            # Create TLV dataset
            dataset = TLVDataset("my_tlv_dataset")

            # Process frames
            for frame in dataset:
                point_cloud = frame.output.pointcloud
                print(f"Frame: {frame.timestamp}, Points: {len(point_cloud)}")
    """

    def __init__(self, datadir: str, config: Optional[ConfigFile] = None) -> None:
        """Initialize TLV dataset.

        Args:
            datadir: Directory path containing TLV data files or dataset name
            config: Optional radar configuration
        """
        # Resolve dataset name to absolute path if needed
        if not os.path.isabs(datadir) or not os.path.isdir(datadir):
            dataset_index = _try_import_dataset_index()
            if dataset_index is not None:
                for base in get_all_base_dirs("TLV"):
                    resolved = dataset_index.resolve_recording_path(datadir, base)
                    if resolved:
                        datadir = resolved
                        break
        super().__init__(datadir, config)
        self._frames = self._load_data()

    def _load_data(self) -> List[Frame]:
        """Load frames from the dataset directory.

        Returns:
            List of Frame objects
        """
        try:
            # Check if directory exists
            if not os.path.exists(self.dataset_path):
                logger.warning(f"Dataset directory not found: {self.dataset_path}")
                return []

            # Load frames using read_directory
            frames = list(read_directory(self.dataset_path, config=self.config))
            logger.info(f"Loaded {len(frames)} frames from {self.dataset_path}")
            return frames
        except Exception as e:
            logger.warning(f"Error loading dataset: {e}")
            return []

    def __iter__(self):
        """Iterate over frames in the dataset."""
        return iter(self._frames)

    def __len__(self):
        """Get the number of frames in the dataset."""
        return len(self._frames)

    @functools.cached_property
    def labels(self) -> Optional["Labels"]:
        """Load labels if a labels/ directory or CSV files exist.

        Returns:
            Labels instance, or None if no labels are found.
        """
        from vt_radar.data.labels import Labels

        labels_dir = os.path.join(self.dataset_path, "labels")
        if os.path.isdir(labels_dir):
            return Labels.load(self.dataset_path)
        # Legacy: check for CSV in dataset root
        csv_files = glob.glob(os.path.join(self.dataset_path, "*.csv"))
        if csv_files:
            return Labels.load(self.dataset_path)
        return None

    @property
    def timestamps(self) -> pd.Series:
        """Series of all frame timestamps for alignment."""
        return pd.Series([f.timestamp for f in self._frames])


class RawDataIterator:
    """Iterator for raw radar ADC data.

    This iterator provides access to individual frames from raw radar
    ADC data arrays.

    Attributes:
        _data (np.ndarray): Raw radar data array
        _idx (int): Current frame index

    Example:
        .. code-block:: python

            # Create iterator for raw data
            iterator = RawDataIterator(raw_adc_data)

            # Iterate through frames
            for frame in iterator:
                print(f"Frame shape: {frame._radar.shape}")
    """

    def __init__(self, data: np.ndarray) -> None:
        """Initialize the raw data iterator.

        Args:
            data: Raw radar ADC data array
        """
        self._data = data
        self._idx = 0

    def __iter__(self) -> 'RawDataIterator':
        """Return the iterator object."""
        return self
 
    def __next__(self) -> Frame:
        """Get the next frame from the raw data.

        Returns:
            Frame: Next frame with radar data set

        Raises:
            StopIteration: When all frames have been processed
        """
        if self._idx >= len(self._data):
            raise StopIteration
        f = Frame()
        f.set_radar(self._data[self._idx, :, :, :])
        self._idx += 1
        return f

class RawDataset(ValtesDataset):
    def __init__(self, datadir):
        super().__init__(datadir, type_="DCA1000")
    
    def __len__(self):
        return len(self._all_data)
    
    def __iter__(self) -> Frame:
        return RawDataIterator(self._all_data)
    
    def _read_data(self):
        self._data_files = self.get_directory_files(self._dir, "*.bin")
        config = self.config
        sensor = config.sensor()
        for f in self._data_files:
            # the int16 is the matlab scripts in TI documentation
            # openradar had uint16 bit this created overflows in ADC samples (but good point clouds)
            adc_data = np.fromfile(f, dtype=np.int16)
#            adc_data = adc_data - (adc_data >= 2**15) * 2**16

            print(sensor.num_chirps_per_frame, sensor.num_rx_antennas, sensor.num_adc_samples)
            est_frame_size  =sensor.num_chirps_per_frame * sensor.num_rx_antennas * sensor.num_adc_samples
            print(est_frame_size)

            try:
                adc_data = adc_data.reshape((-1,
                                            2* #x2 because complex
                                            sensor.num_chirps_per_frame *
                                            sensor.num_rx_antennas *
                                            sensor.num_adc_samples))
            except Exception as e:
                print(sensor.num_chirps_per_frame, sensor.num_rx_antennas, sensor.num_adc_samples)
                est_frame_size  =sensor.num_chirps_per_frame * sensor.num_rx_antennas * sensor.num_adc_samples
                print(est_frame_size)
                print(adc_data.shape[0] / float(est_frame_size))
                print(e)
                raise
            print(adc_data.shape)
            all_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=sensor.num_chirps_per_frame,
                               num_rx=sensor.num_rx_antennas, num_samples=sensor.num_adc_samples)

            self._all_data = all_data

class RawDataIterator2:
    def __init__(self, h5):
        self._h5 = h5
        self._idx = 0

    def __iter__(self):
        return self
 
    def __next__(self):
        # Stop when the frame streams are exhausted. Without this bound the
        # iterator never raises StopIteration and instead ran off the end into
        # an IndexError (and it printed the stream count on every call).
        if self._idx >= len(self._h5["radar"]):
            raise StopIteration
        f = Frame()
        img = self._h5["rgb"][self._idx]
        rad = self._h5["radar"][self._idx]
        if img is not None:
            f.set_image(img)
        f.set_radar(rad)
        self._idx += 1
        return f

class RawDataset2(ValtesDataset):
    def __init__(self, datadir):
        super().__init__(datadir, type_="DCA1000")

    def _load_single_h5(self, filename):
        h5 = h5py.File(filename, 'r')
        streams_available = list(h5.keys())
        dataset_len = len(h5[streams_available[0]])
        return h5, streams_available, dataset_len

    def __len__(self):
        h5, streams_available, dataset_len = self._load_single_h5(self._data_files[0])
        return dataset_len
    
    def __iter__(self) -> Frame:
        h5, streams_available, dataset_len = self._load_single_h5(self._data_files[0])
        return RawDataIterator2(h5)
    
    def __pieter__(self) -> Frame:
        h5 = {}
        f = Frame()
        idx = 0
        img = h5["rgb"][idx]
        rad = h5["radar"][idx]
        f.set_image(img)
        f.set_radar(rad)
        yield f

def list_datasets(type_: str = "TLV") -> List[str]:
    """List available datasets of a specific type.

    Supports both flat and YYMM-grouped directory layouts. Tries index.json
    first for speed, falls back to directory scanning.

    Args:
        type_: Dataset type to list (e.g., "TLV", "DCA1000")

    Returns:
        List[str]: List of dataset directory names (IDs)

    Example:
        .. code-block:: python

            # List all TLV datasets
            datasets = list_datasets("TLV")
            print(f"Available TLV datasets: {datasets}")

            # List all raw datasets
            raw_datasets = list_datasets("DCA1000")
    """
    dataset_index = _try_import_dataset_index()
    if dataset_index is None:
        # vt_radar.tools (dataset index) is not shippable with the recorder
        # subset — dataset listing is a toolbox-only convenience, so return
        # an empty result rather than crashing.
        return []
    load_index = dataset_index.load_index
    import re
    yymm_pattern = re.compile(r"^\d{4}$")

    seen: set = set()
    datasets: List[str] = []

    for base_dir in get_all_base_dirs(type_):
        if not os.path.isdir(base_dir):
            continue

        # Fast path: read from index.json
        index = load_index(base_dir)
        if index:
            for ds in index.datasets:
                ds_id = ds["id"]
                if ds_id not in seen:
                    seen.add(ds_id)
                    datasets.append(ds_id)
            continue

        # Fallback: scan flat dirs + recurse into YYMM dirs
        for entry in sorted(os.listdir(base_dir)):
            entry_path = os.path.join(base_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if yymm_pattern.match(entry):
                for sub in sorted(os.listdir(entry_path)):
                    sub_path = os.path.join(entry_path, sub)
                    if os.path.isdir(sub_path) and glob.glob(os.path.join(sub_path, "*.h5")):
                        if sub not in seen:
                            seen.add(sub)
                            datasets.append(sub)
            elif glob.glob(os.path.join(entry_path, "*.h5")):
                if entry not in seen:
                    seen.add(entry)
                    datasets.append(entry)

    return datasets