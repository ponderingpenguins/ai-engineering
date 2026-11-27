"""Base classes for consumer data in the radar processing pipeline.

This module provides the base ConsumerData class and related utilities
for structured data storage in Frame objects.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
import json
import numpy as np

@dataclass
class ConsumerData(ABC):
    """Base class for all consumer data.
    
    All consumer data classes should inherit from this base class
    and implement the serialize() and deserialize() methods.
    
    Attributes:
        consumer_name: Name of the consumer that created this data
        timestamp: When this data was created
        serialized: Whether this data is already serialized for JSON
    """
    consumer_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    serialized: bool = False
    
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize this data to a JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: Serialized data ready for JSON transmission
        """
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ConsumerData':
        """Deserialize data from a dictionary.
        
        Args:
            data: Dictionary containing serialized data
            
        Returns:
            ConsumerData: Deserialized consumer data object
        """
        pass
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary with metadata.
        
        Returns:
            Dict[str, Any]: Complete serialized data with metadata
        """
        return {
            'consumer_name': self.consumer_name,
            'timestamp': self.timestamp.isoformat(),
            'serialized': self.serialized,
            'data': self.serialize()
        }
    
    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> 'ConsumerData':
        """Create from JSON dictionary with metadata.
        
        Args:
            data: Dictionary containing serialized data with metadata
            
        Returns:
            ConsumerData: Deserialized consumer data object
        """
        # Extract metadata
        consumer_name = data.get('consumer_name', 'unknown')
        timestamp_str = data.get('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        serialized = data.get('serialized', False)
        
        # Create instance
        instance = cls.deserialize(data['data'])
        instance.consumer_name = consumer_name
        instance.timestamp = timestamp
        instance.serialized = serialized
        
        return instance

class JSONDict(dict):
    """Immutable marker subclass of dict to represent JSON-ready objects."""

    def __readonly(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is immutable")

    # Block mutating methods
    __setitem__ = __readonly
    __delitem__ = __readonly
    clear       = __readonly
    pop         = __readonly
    popitem     = __readonly
    setdefault  = __readonly
    update      = __readonly


class JSONList(list):
    """Immutable marker subclass of list to represent JSON-ready arrays."""

    def __readonly(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is immutable")

    # Block mutating methods
    __setitem__ = __readonly
    __delitem__ = __readonly
    append      = __readonly
    extend      = __readonly
    insert      = __readonly
    pop         = __readonly
    remove      = __readonly
    clear       = __readonly
    sort        = __readonly
    reverse     = __readonly


def serialize_numpy_array(arr: np.ndarray) -> Any:
    """Helper function to serialize numpy arrays with robust edge case handling.
    
    This function handles all the edge cases that listify() used to handle:
    - None arrays
    - Empty arrays
    - Scalar arrays
    - Infinity and NaN values
    - Different numpy data types
    
    Args:
        arr: Numpy array to serialize
        
    Returns:
        Any: Serialized array (list or scalar) ready for JSON transmission
    """
    if arr is None:
        return None
    if arr.size == 0:
        return []
    if arr.ndim == 0:
        # Handle scalar arrays. arr.item() returns a *Python* native (not a
        # numpy scalar), so the old isinstance(item, np.float64/np.int_/...)
        # checks never matched and inf/nan leaked through as invalid JSON.
        # Match the built-in types and scrub non-finite values to the same
        # sentinels the multi-dim path uses via nan_to_num.
        item = arr.item()
        if isinstance(item, bool):  # bool is a subclass of int — check first
            return bool(item)
        elif isinstance(item, int):
            return int(item)
        elif isinstance(item, float):
            if np.isnan(item):
                return 1e9
            if np.isinf(item):
                return 1e9 if item > 0 else -1e9
            return float(item)
        else:
            return item
    
    # Handle multi-dimensional arrays
    # Replace inf and nan with large numbers (same as listify)
    arr_clean = np.nan_to_num(arr, nan=1e9, posinf=1e9, neginf=-1e9)
    return JSONList(arr_clean.tolist())


def deserialize_numpy_array(data: Any, dtype: np.dtype = None) -> Optional[np.ndarray]:
    """Helper function to deserialize numpy arrays.
    
    Args:
        data: Serialized array data
        dtype: Optional numpy dtype to use
        
    Returns:
        Optional[np.ndarray]: Deserialized numpy array or None
    """
    if data is None or data == []:
        return None
    if isinstance(data, (int, float)):
        return np.array([data], dtype=dtype)
    return np.array(data, dtype=dtype)


def serialize_for_json(data: Any) -> Any:
    """Universal serialization function for JSON transmission.
    
    This function replaces the old listify() function and provides
    a unified approach to serializing any data type for JSON transmission.
    
    Args:
        data: Any data to serialize

    Returns:
        Any: JSON-serializable data
    """

    @dataclass
    class Serializer:
        visited: set = field(default_factory=set)

        def serialize_for_json(self, data: Any) -> Any:
            if isinstance(data, (JSONDict, JSONList, bytes)):
                return data
            elif isinstance(data, dict):
                data_id = id(data)
                if data_id in self.visited:
                    return "[Circular Reference]"

                self.visited.add(data_id)
                try:
                    result = JSONDict({key: self.serialize_for_json(value) for key, value in data.items()})
                finally:
                    self.visited.discard(data_id)

                return result
            elif isinstance(data, (list, tuple)):
                data_id = id(data)
                if data_id in self.visited:
                    return "[Circular Reference]"

                self.visited.add(data_id)
                try:
                    result = JSONList([self.serialize_for_json(item) for item in data])
                finally:
                    self.visited.discard(data_id)

                return result
            elif isinstance(data, np.ndarray):
                return serialize_numpy_array(data)
            elif isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(data)
            elif isinstance(data, (np.float16, np.float32, np.float64)):
                if np.isinf(data):
                    return 1e9 if data > 0 else -1e9
                elif np.isnan(data):
                    return 1e9  # Convert NaN to large number (as the py-float branch does)
                return float(data)
            elif isinstance(data, np.bool_):
                return bool(data)
            elif isinstance(data, float):
                if data == float('inf') or data == float('-inf'):
                    return 1e9 if data > 0 else -1e9
                elif np.isnan(data):
                    return 1e9  # Convert NaN to large number
                return data
            else:
                return data

    serializer = Serializer()
    return serializer.serialize_for_json(data)
