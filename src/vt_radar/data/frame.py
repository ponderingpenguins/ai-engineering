"""Frame module for radar data frame representation.

This module provides the Frame class, which is the core data structure
for representing individual radar frames with associated metadata,
processed data, and optional image data.
"""

import numpy as np
from typing import Any, Dict, Optional, Type, TypeVar, Union, overload

from vt_radar.data.image import Image
from vt_radar.parsing_utils.parseFrame import parseStandardFrame
from vt_radar.data.output import ProcessedRadarData
from vt_radar.config.ti_config import ConfigFile
from vt_radar.data.consumer_data import ConsumerData

T = TypeVar("T", bound=ConsumerData)

# Mock parseTLVs function for testing
def parseTLVs(data, config):
    """Mock parseTLVs function for testing."""
    return {"parsed": "data"}


class Frame(object):
    """Core data structure for radar frames.

    The Frame class represents a single radar frame containing raw TLV data,
    processed output, optional image data, and metadata. It serves as the
    primary data structure for radar data processing and analysis.

    Attributes:
        _timestamp (Optional[datetime]): Frame timestamp
        _tlvs (Optional[bytes]): Raw TLV data from radar
        _image (Optional[bytes]): Raw JPEG image data
        _rgb_image (Optional[np.ndarray]): Decoded RGB image data
        _config (Optional[ConfigFile]): Radar configuration
        _output (Optional[ProcessedRadarData]): Processed radar output data
        _radar (Optional[np.ndarray]): Raw radar ADC data

    Example:
        Create a frame from raw data:

        .. code-block:: python

            from vt_radar.data import Frame
            from vt_radar.config.ti_config import ConfigFile

            # Load configuration
            config = ConfigFile()
            config.parseCfg('path/to/config.cfg')

            # Create frame
            frame = Frame(
                timestamp=datetime.now(),
                tlvs=raw_tlv_data,
                image=raw_jpeg_data,
                config=config
            )

            # Access processed data
            point_cloud = frame.output.pointcloud
            track_data = frame.output.trackdata
            rgb_image = frame.image
    """

    def __init__(self, timestamp: Optional[Any] = None, tlvs: Optional[bytes] = None, 
                 image: Optional[bytes] = None, config: Optional[ConfigFile] = None) -> None:
        """Initialize a radar frame.

        Args:
            timestamp: Frame timestamp (datetime object)
            tlvs: Raw TLV data from radar device
            image: Raw JPEG image data (bytes)
            config: Radar configuration object
            
        Raises:
            FrameError: If invalid data is provided
        """
        # Validate timestamp - only allow datetime objects or None
        if timestamp is not None and not hasattr(timestamp, 'year'):
            from vt_radar.exceptions import FrameError
            raise FrameError(f"Invalid timestamp type: {type(timestamp)}")
        
        # Validate TLV data
        if tlvs is not None and not isinstance(tlvs, (bytes, bytearray)):
            from vt_radar.exceptions import FrameError
            raise FrameError(f"Invalid TLV data type: {type(tlvs)}")
        
        self._timestamp = timestamp
        self._tlvs = tlvs
        self._image: Optional[bytes] = bytes(image) if image is not None else None  # raw jpeg bytes
        self._rgb_image: Optional[np.ndarray] = None
        self._config = config
        self._output: Optional[ProcessedRadarData] = None
        self._radar: Optional[np.ndarray] = None
        
        # Consumer data storage - key-value store for consumer data
        self._consumer_data: Dict[str, ConsumerData] = {}
    
    @staticmethod
    def from_raw(data, config: ConfigFile) -> 'Frame':
        """Create a Frame instance from raw data.

        Args:
            data: Either a dictionary containing frame data or raw TLV bytes
            config: Radar configuration object

        Returns:
            Frame: New Frame instance with data from the input

        Example:
            .. code-block:: python

                # Create frame from raw TLV data
                frame = Frame.from_raw(tlv_bytes, config)
                
                # Create frame from dataset row
                frame = Frame.from_raw({
                    'timestamp': datetime.now(),
                    'frame': tlv_bytes,
                    'image': jpeg_bytes,
                    'mmwave': adc_data
                }, config)
        """
        if isinstance(data, dict):
            # Dictionary format
            timestamp = data["timestamp"]
            frame = data.get("frame", None)  # TLVs
            image = data.get("image", None)  # raw jpeg
            adc = data.get("mmwave", None)  # raw
        else:
            # Raw TLV data format
            timestamp = None
            frame = data  # TLVs
            image = None
            adc = None
            
        # Create Frame object first
        frame_obj = Frame(timestamp, frame, image, config)
        
        # Parse TLV data if available
        if frame is not None:
            parsed_data = parseTLVs(frame, config)
            # Store parsed data for later use
            frame_obj._parsed_data = parsed_data
            
        if adc is not None:
            frame_obj.set_radar(adc)
        return frame_obj

    def __str__(self) -> str:
        """String representation of the frame.

        Returns:
            str: Human-readable string describing the frame content
        """
        timestamp_str = str(self._timestamp) if self._timestamp else "No timestamp"
        if self._output:
            return f"Frame({timestamp_str}) {str(self.output)} has_image={self._image is not None}"
        if self._radar is not None:
            return f"Frame({timestamp_str}) {str(self._radar.shape)} has_image={self._image is not None}"
        return f"Frame({timestamp_str})"

    @property
    def frame(self) -> Optional[bytes]:
        """Get the raw TLV frame data.

        Returns:
            Optional[bytes]: Raw TLV data from the radar device, or None if not available
        """
        return self._tlvs
    
    @property
    def raw_image(self) -> Optional[bytes]:
        """Get the raw image data.
        
        Returns:
            Optional[str]: Raw image data (bytes)
        """
        return self._image

    @property
    def image(self) -> Image:
        """Get the decoded RGB image data.

        This property lazily decodes the JPEG image data on first access.
        The decoded image is cached for subsequent accesses.

        Returns:
            Image: Decoded RGB image data, or empty Image if no image data available

        Note:
            The image is automatically converted from BGR to RGB format
            for proper display with matplotlib and other visualization tools.
        """
        if self._image is not None:
            if self._rgb_image is None:
                try:
                    import cv2
                    # would be nicer to have this in Image class
                    jpeg_image = np.frombuffer(self._image, dtype=np.uint8)
                    image = cv2.imdecode(jpeg_image, cv2.IMREAD_COLOR)
                    if image is not None:
                        self._rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self._rgb_image = Image(self._rgb_image)
                    else:
                        # Invalid JPEG data, return empty image
                        self._rgb_image = Image(np.array([]))
                except Exception:
                    # Error decoding image, return empty image
                    self._rgb_image = Image(np.array([]))
            return self._rgb_image
        return Image(None)
    
    @property
    def jpg(self) -> Optional[np.ndarray]:
        """Get the raw JPEG image data as numpy array.

        Returns:
            Optional[np.ndarray]: Raw JPEG data as uint8 array, or None if no image data available
        """
        if self._image is not None:
            jpeg_image = np.frombuffer(self._image, dtype=np.uint8)
            return jpeg_image
        return None
    
    def _parse_tlvs(self) -> ProcessedRadarData:
        """Parse TLV data and create ProcessedRadarData object.

        This is an internal method that parses the raw TLV data using
        the radar configuration to create a structured ProcessedRadarData object.

        Returns:
            ProcessedRadarData: Parsed radar output data

        Note:
            This method is called automatically when the output property
            is accessed for the first time.
        """
        from vt_radar.parsing_utils.parseFrame import parseStandardFrame
        raw_data = parseStandardFrame(self._tlvs, self._config)
        output = ProcessedRadarData(raw_data=raw_data)
        return output

    @property
    def output(self) -> ProcessedRadarData:
        """Get the processed radar output data.

        This property lazily parses the TLV data on first access and
        caches the result for subsequent accesses.

        Returns:
            ProcessedRadarData: Processed radar output containing point clouds,
                track data, and other radar processing results

        Raises:
            RuntimeError: If TLV data is not available or parsing fails
        """
        if self._output is None:
            self._output = self._parse_tlvs()

        return self._output
    
    @output.setter
    def output(self, new_output: ProcessedRadarData) -> None:
        """Set the processed radar output data.

        Args:
            new_output: New ProcessedRadarData object to set
        """
        self._output = new_output
    
    def output2(self) -> dict:
        """Get raw output dictionary without ProcessedRadarData object wrapper.

        Returns:
            dict: Raw output dictionary from TLV parsing

        Note:
            This method provides direct access to the parsed dictionary
            without the ProcessedRadarData object wrapper. Use the output property
            for the recommended interface.
        """
        from vt_radar.parsing_utils.parseFrame import parseStandardFrame
        raw_data = parseStandardFrame(self._tlvs, self._config)
        return raw_data.to_dict()
    
    @property
    def timestamp(self) -> Optional[Any]:
        """Get the frame timestamp.

        Returns:
            Optional[Any]: Frame timestamp, or None if not set
        """
        return self._timestamp
    
    @property
    def frame_num(self) -> int:
        """Get the frame number.

        Returns:
            int: Frame number from the processed radar data
        """
        if self._output is None:
            self._output = self._parse_tlvs()
        return self._output.frame_num
    
    def set_image(self, image: str) -> None:
        """Set the raw JPEG image data.

        Args:
            image: Raw JPEG image data (bytes)

        Note:
            This will invalidate any cached RGB image data.
        """
        self._image = image
        # Invalidate the decoded-image cache; the `image` property lazily
        # JPEG-decodes _image on next access. Eagerly calling Image(image) here
        # passed raw JPEG bytes straight into Image() (DataError) instead of
        # decoding them.
        self._rgb_image = None

    def set_radar(self, radar: np.ndarray) -> None:
        """Set the raw radar ADC data.

        Args:
            radar: Raw radar ADC data array
        """
        self._radar = radar 
    
    def __lshift__(self, consumer_data: ConsumerData) -> 'Frame':
        """Left shift operator for adding consumer data to the frame.
        
        This allows consumers to add their data using the << operator:
        frame << RadarProcessingData(...)
        
        Args:
            consumer_data: ConsumerData object to add to the frame
            
        Returns:
            Frame: Self for method chaining
            
        Example:
            frame << RadarProcessingData(consumer_name="RadarProcessing", ...)
        """
        if not isinstance(consumer_data, ConsumerData):
            raise TypeError(f"Expected ConsumerData, got {type(consumer_data)}")
        
        self._consumer_data[consumer_data.consumer_name] = consumer_data
        return self
    
    @overload
    def __rshift__(self, key: str) -> Optional[ConsumerData]: ...
    @overload
    def __rshift__(self, key: Type[T]) -> Optional[T]: ...

    def __rshift__(self, key: Union[str, Type[T]]) -> Optional[ConsumerData]:
        """Right shift operator for getting consumer data from the frame.

        Two forms are supported:
        - ``frame >> "RadarProcessing"`` — legacy string lookup by consumer_name.
        - ``frame >> RadarProcessingData``  — typed lookup by class; mypy
          narrows the return to ``Optional[RadarProcessingData]``.

        The class form has zero knowledge leakage into Frame: any
        ``ConsumerData`` subclass works without registering the name here.

        Args:
            key: Either the consumer_name string or a ConsumerData subclass.

        Returns:
            The matching ConsumerData instance, or None if absent / wrong type.
        """
        if isinstance(key, type):
            for data in self._consumer_data.values():
                if isinstance(data, key):
                    return data
            return None
        return self._consumer_data.get(key)
    
    def add_consumer_data(self, consumer_data: ConsumerData) -> None:
        """Add consumer data to the frame.
        
        Args:
            consumer_data: ConsumerData object to add to the frame
        """
        if not isinstance(consumer_data, ConsumerData):
            raise TypeError(f"Expected ConsumerData, got {type(consumer_data)}")
        
        self._consumer_data[consumer_data.consumer_name] = consumer_data
    
    def get_consumer_data(self, consumer_name: str) -> Optional[ConsumerData]:
        """Get consumer data by consumer name.
        
        Args:
            consumer_name: Name of the consumer
            
        Returns:
            Optional[ConsumerData]: Consumer data or None if not found
        """
        return self._consumer_data.get(consumer_name)
    
    def get_all_consumer_data(self, serialized_only: bool = False) -> Dict[str, ConsumerData]:
        """Get all consumer data.
        
        Args:
            serialized_only: If True, only return serialized data
            
        Returns:
            Dict[str, ConsumerData]: Dictionary of consumer data
        """
        if serialized_only:
            return {k: v for k, v in self._consumer_data.items() if v.serialized}
        return self._consumer_data.copy()
    
    def get_serialized_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all consumer data as serialized dictionaries.
        
        Returns:
            Dict[str, Dict[str, Any]]: Serialized consumer data ready for JSON
        """
        return {k: v.to_json_dict() for k, v in self._consumer_data.items()}
    
    def has_consumer_data(self, consumer_name: str) -> bool:
        """Check if consumer data exists for a given consumer.
        
        Args:
            consumer_name: Name of the consumer
            
        Returns:
            bool: True if consumer data exists
        """
        return consumer_name in self._consumer_data
    
    def remove_consumer_data(self, consumer_name: str) -> bool:
        """Remove consumer data for a given consumer.
        
        Args:
            consumer_name: Name of the consumer
            
        Returns:
            bool: True if data was removed, False if not found
        """
        if consumer_name in self._consumer_data:
            del self._consumer_data[consumer_name]
            return True
        return False
    
    def serialize_image_data(self) -> Optional[Dict[str, Any]]:
        """Serialize image data for frontend.

        Returns:
            Optional[Dict[str, Any]]: Serialized image data or None if no image
        """
        if self._image is not None:
            return {
                'data': self._image,  # JPEG data bytes object is encoded as Base64
                'format': 'jpeg'
            }
        return None
    
    def serialize_heatmap_data(self) -> Optional[Dict[str, Any]]:
        """Serialize heatmap data for frontend.
        
        Returns:
            Optional[Dict[str, Any]]: Serialized heatmap data or None if no heatmap
        """
        if hasattr(self.output, 'range_doppler_heatmap') and self.output.range_doppler_heatmap is not None:
            heatmap_data = self.output.range_doppler_heatmap
            if hasattr(heatmap_data, 'data') and heatmap_data.data is not None:
                return {
                    'rangeDopplerHeatmap': heatmap_data.data.tolist(),
                    'width': heatmap_data.data.shape[1] if heatmap_data.data.ndim >= 2 else 64,
                    'height': heatmap_data.data.shape[0] if heatmap_data.data.ndim >= 1 else 16,
                    'minValue': float(np.min(heatmap_data.data)),
                    'maxValue': float(np.max(heatmap_data.data))
                }
        return None
    
    def serialize_label_data(self) -> Optional[Dict[str, Any]]:
        """Serialize label data for frontend.
        
        Returns:
            Optional[Dict[str, Any]]: Serialized label data or None if no label
        """
        if hasattr(self, 'label') and self.label is not None:
            return {
                'label': self.label,
                'frameNum': getattr(self, 'frame_num', 0),
                'confidence': getattr(self, 'label_confidence', 1.0),
                'labelMap': getattr(self, 'label_map', {})
            }
        return None
