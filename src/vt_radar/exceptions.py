"""Exception classes for the Valtes Radar Toolbox.

This module provides a comprehensive set of exception classes for handling
errors throughout the radar toolbox. It follows a hierarchical structure
that allows for both specific error handling and general error catching.

Exception Hierarchy:
    RadarToolboxError (Base)
    ├── IntakeError
    │   ├── RadarIntakeError
    │   ├── DatasetIntakeError
    │   └── WebcamError
    ├── ProcessingError
    │   ├── FrameProcessingError
    │   ├── TrackingError
    │   └── ClusteringError
    ├── DataError
    │   ├── FrameError
    │   ├── DatasetError
    │   └── ConfigurationError
    ├── FrameworkError
    │   ├── ProducerError
    │   ├── ConsumerError
    │   └── QueueError
    └── HardwareError
        ├── RadarError
        ├── ComPortError
        └── CameraError

Example:
    Basic exception handling:

    .. code-block:: python

        from vt_radar.exceptions import RadarIntakeError, IntakeError

        try:
            radar_intake.start()
        except RadarIntakeError as e:
            logger.error(f"Radar intake failed: {e}")
            # Handle specific radar intake error
        except IntakeError as e:
            logger.error(f"General intake error: {e}")
            # Handle any intake error
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            # Handle unexpected errors
"""

from typing import Optional, Dict, Any


class RadarToolboxError(Exception):
    """Base exception class for all radar toolbox errors.

    This is the root exception class that all other exceptions in the
    radar toolbox inherit from. It provides common functionality for
    error handling and logging.

    Attributes:
        message (str): Error message describing what went wrong
        error_code (Optional[str]): Optional error code for programmatic handling
        details (Optional[Dict[str, Any]]): Additional error details
        original_error (Optional[Exception]): The original exception that caused this error
    """

    def __init__(self, message: str, error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 original_error: Optional[Exception] = None) -> None:
        """Initialize the radar toolbox error.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Additional error details as a dictionary
            original_error: The original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation of the error."""
        base_msg = f"{self.__class__.__name__}: {self.message}"
        if self.error_code:
            base_msg += f" (Code: {self.error_code})"
        if self.details:
            base_msg += f" (Details: {self.details})"
        return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary for serialization.

        Returns:
            Dictionary representation of the error
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'original_error': str(self.original_error) if self.original_error else None
        }


# Intake-related exceptions
class IntakeError(RadarToolboxError):
    """Base exception for intake-related errors.

    This exception is raised when there are problems with data intake
    from various sources (radar, dataset, webcam, etc.).
    """

    def __init__(self, message: str, source: Optional[str] = None, **kwargs) -> None:
        """Initialize intake error.

        Args:
            message: Error message
            source: Source of the intake error (e.g., 'radar', 'dataset', 'webcam')
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, **kwargs)
        self.source = source


class RadarIntakeError(IntakeError):
    """Exception raised for radar-specific intake errors.

    This exception is raised when there are problems with radar data
    intake, such as COM port connection issues, configuration problems,
    or UART communication failures.
    """

    def __init__(self, message: str, com_port: Optional[str] = None, **kwargs) -> None:
        """Initialize radar intake error.

        Args:
            message: Error message
            com_port: COM port that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, source='radar', **kwargs)
        self.com_port = com_port


class DatasetIntakeError(IntakeError):
    """Exception raised for dataset-specific intake errors.

    This exception is raised when there are problems with dataset
    intake, such as file not found, corrupted data, or format issues.
    """

    def __init__(self, message: str, dataset_path: Optional[str] = None, **kwargs) -> None:
        """Initialize dataset intake error.

        Args:
            message: Error message
            dataset_path: Path to the dataset that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, source='dataset', **kwargs)
        self.dataset_path = dataset_path


class WebcamError(IntakeError):
    """Exception raised for webcam-specific intake errors.

    This exception is raised when there are problems with webcam
    intake, such as camera not found, capture failures, or OpenCV errors.
    """

    def __init__(self, message: str, camera_port: Optional[int] = None, **kwargs) -> None:
        """Initialize webcam error.

        Args:
            message: Error message
            camera_port: Camera port that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, source='webcam', **kwargs)
        self.camera_port = camera_port


# Processing-related exceptions
class ProcessingError(RadarToolboxError):
    """Base exception for processing-related errors.

    This exception is raised when there are problems with data
    processing, such as frame processing, tracking, or clustering.
    """

    def __init__(self, message: str, processing_stage: Optional[str] = None, **kwargs) -> None:
        """Initialize processing error.

        Args:
            message: Error message
            processing_stage: Stage of processing that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, **kwargs)
        self.processing_stage = processing_stage


class FrameProcessingError(ProcessingError):
    """Exception raised for frame processing errors.

    This exception is raised when there are problems with individual
    frame processing, such as TLV parsing, frame validation, or
    data extraction.
    """

    def __init__(self, message: str, frame_id: Optional[int] = None, **kwargs) -> None:
        """Initialize frame processing error.

        Args:
            message: Error message
            frame_id: ID of the frame that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, processing_stage='frame', **kwargs)
        self.frame_id = frame_id


class TrackingError(ProcessingError):
    """Exception raised for tracking-related errors.

    This exception is raised when there are problems with object
    tracking, such as track initialization, update failures, or
    track association errors.
    """

    def __init__(self, message: str, track_id: Optional[int] = None, **kwargs) -> None:
        """Initialize tracking error.

        Args:
            message: Error message
            track_id: ID of the track that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, processing_stage='tracking', **kwargs)
        self.track_id = track_id


class ClusteringError(ProcessingError):
    """Exception raised for clustering-related errors.

    This exception is raised when there are problems with point
    cloud clustering, such as DBSCAN failures, parameter issues,
    or cluster validation errors.
    """

    def __init__(self, message: str, cluster_id: Optional[int] = None, **kwargs) -> None:
        """Initialize clustering error.

        Args:
            message: Error message
            cluster_id: ID of the cluster that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, processing_stage='clustering', **kwargs)
        self.cluster_id = cluster_id


# Data-related exceptions
class DataError(RadarToolboxError):
    """Base exception for data-related errors.

    This exception is raised when there are problems with data
    structures, datasets, or data validation.
    """

    def __init__(self, message: str, data_type: Optional[str] = None, **kwargs) -> None:
        """Initialize data error.

        Args:
            message: Error message
            data_type: Type of data that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, **kwargs)
        self.data_type = data_type


class FrameError(DataError):
    """Exception raised for frame-related errors.

    This exception is raised when there are problems with frame
    data structures, such as invalid frame format, missing data,
    or corruption.
    """

    def __init__(self, message: str, frame_id: Optional[int] = None, **kwargs) -> None:
        """Initialize frame error.

        Args:
            message: Error message
            frame_id: ID of the frame that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, data_type='frame', **kwargs)
        self.frame_id = frame_id


class DatasetError(DataError):
    """Exception raised for dataset-related errors.

    This exception is raised when there are problems with datasets,
    such as file access issues, format problems, or data corruption.
    """

    def __init__(self, message: str, dataset_path: Optional[str] = None, **kwargs) -> None:
        """Initialize dataset error.

        Args:
            message: Error message
            dataset_path: Path to the dataset that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, data_type='dataset', **kwargs)
        self.dataset_path = dataset_path


class ConfigurationError(DataError):
    """Exception raised for configuration-related errors.

    This exception is raised when there are problems with configuration
    files, such as invalid parameters, missing settings, or format issues.
    """

    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs) -> None:
        """Initialize configuration error.

        Args:
            message: Error message
            config_file: Path to the configuration file that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, data_type='configuration', **kwargs)
        self.config_file = config_file


# Framework-related exceptions
class FrameworkError(RadarToolboxError):
    """Base exception for framework-related errors.

    This exception is raised when there are problems with the
    producer-consumer framework, such as thread issues, queue
    problems, or synchronization errors.
    """

    def __init__(self, message: str, component: Optional[str] = None, **kwargs) -> None:
        """Initialize framework error.

        Args:
            message: Error message
            component: Framework component that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, **kwargs)
        self.component = component


class ProducerError(FrameworkError):
    """Exception raised for producer-related errors.

    This exception is raised when there are problems with data
    producers, such as production failures, thread issues, or
    data generation errors.
    """

    def __init__(self, message: str, producer_name: Optional[str] = None, **kwargs) -> None:
        """Initialize producer error.

        Args:
            message: Error message
            producer_name: Name of the producer that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, component='producer', **kwargs)
        self.producer_name = producer_name


class ConsumerError(FrameworkError):
    """Exception raised for consumer-related errors.

    This exception is raised when there are problems with data
    consumers, such as processing failures, thread issues, or
    data consumption errors.
    """

    def __init__(self, message: str, consumer_name: Optional[str] = None, **kwargs) -> None:
        """Initialize consumer error.

        Args:
            message: Error message
            consumer_name: Name of the consumer that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, component='consumer', **kwargs)
        self.consumer_name = consumer_name


class QueueError(FrameworkError):
    """Exception raised for queue-related errors.

    This exception is raised when there are problems with queues,
    such as overflow, underflow, or synchronization issues.
    """

    def __init__(self, message: str, queue_name: Optional[str] = None, **kwargs) -> None:
        """Initialize queue error.

        Args:
            message: Error message
            queue_name: Name of the queue that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, component='queue', **kwargs)
        self.queue_name = queue_name


# Hardware-related exceptions
class HardwareError(RadarToolboxError):
    """Base exception for hardware-related errors.

    This exception is raised when there are problems with hardware
    devices, such as radar devices, cameras, or communication ports.
    """

    def __init__(self, message: str, device_type: Optional[str] = None, **kwargs) -> None:
        """Initialize hardware error.

        Args:
            message: Error message
            device_type: Type of device that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, **kwargs)
        self.device_type = device_type


class RadarError(HardwareError):
    """Exception raised for radar-specific hardware errors.

    This exception is raised when there are problems with radar
    hardware, such as device initialization, communication failures,
    or configuration issues.
    """

    def __init__(self, message: str, radar_model: Optional[str] = None, **kwargs) -> None:
        """Initialize radar error.

        Args:
            message: Error message
            radar_model: Model of the radar that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, device_type='radar', **kwargs)
        self.radar_model = radar_model


class ComPortError(HardwareError):
    """Exception raised for COM port-related errors.

    This exception is raised when there are problems with COM port
    communication, such as port not found, access denied, or
    communication timeouts.
    """

    def __init__(self, message: str, com_port: Optional[str] = None, **kwargs) -> None:
        """Initialize COM port error.

        Args:
            message: Error message
            com_port: COM port that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, device_type='com_port', **kwargs)
        self.com_port = com_port


class CameraError(HardwareError):
    """Exception raised for camera-related errors.

    This exception is raised when there are problems with camera
    hardware, such as device not found, capture failures, or
    OpenCV initialization errors.
    """

    def __init__(self, message: str, camera_port: Optional[int] = None, **kwargs) -> None:
        """Initialize camera error.

        Args:
            message: Error message
            camera_port: Camera port that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, device_type='camera', **kwargs)
        self.camera_port = camera_port


# AI-related exceptions
class AIError(RadarToolboxError):
    """Base exception for AI-related errors.

    This exception is raised when there are problems with AI algorithms,
    models, or machine learning components.
    """

    def __init__(self, message: str, algorithm: Optional[str] = None, **kwargs) -> None:
        """Initialize AI error.

        Args:
            message: Error message
            algorithm: Name of the algorithm that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(message, **kwargs)
        self.algorithm = algorithm
