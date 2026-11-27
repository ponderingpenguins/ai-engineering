"""Image module for radar image data handling.

This module provides the Image class for handling image data associated
with radar frames, including JPEG decoding and visualization utilities.
"""

import numpy as np
from typing import Optional, Tuple, List, Any

class Image(np.ndarray):
    """Image data structure for radar frame images.

    This class extends numpy's ndarray to provide specialized functionality
    for handling image data associated with radar frames. It supports
    serialization, deserialization, and visualization of image data.

    The Image class is designed to work seamlessly with radar frames that
    may contain synchronized webcam or other image data.

    Attributes:
        chirp_idx (Optional[int]): Chirp index if applicable
        ant_idx (Optional[int]): Antenna index if applicable

    Example:
        Create an image from numpy array:

        .. code-block:: python

            import numpy as np
            from vt_radar.data import Image

            # Create image from array
            image_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            image = Image(image_data)

            # Display the image
            image.plot()

        Create image from serialized data:

        .. code-block:: python

            # Serialize image
            data_list, shape = image.to_list()

            # Recreate image
            restored_image = Image.from_list((data_list, shape))
    """

    def __new__(cls, buffer: Optional[np.ndarray]) -> 'Image':
        """Create a new Image instance.

        Args:
            buffer: Image data as numpy array, or None for empty image

        Returns:
            Image: New Image instance
            
        Raises:
            DataError: If invalid data is provided
        """
        if buffer is None:
            obj = np.array([]).view(cls)
        else:
            # Validate that buffer is a proper numpy array or can be converted
            if not isinstance(buffer, (np.ndarray, list, tuple)) and not hasattr(buffer, '__array__'):
                from vt_radar.exceptions import DataError
                raise DataError(f"Invalid image data type: {type(buffer)}")
            
            try:
                obj = np.asarray(buffer).view(cls)
            except (ValueError, TypeError) as e:
                from vt_radar.exceptions import DataError
                raise DataError(f"Invalid image data: {e}")
        return obj
    
    def to_list(self) -> Tuple[List[float], Tuple[int, ...]]:
        """Convert image to flattened list and shape for serialization.

        Returns:
            Tuple[List[float], Tuple[int, ...]]: A tuple containing:
                - Flattened image data as a list
                - Original shape of the image

        Example:
            .. code-block:: python

                # Serialize image for storage
                data_list, shape = image.to_list()
                # Store data_list and shape for later reconstruction
        """
        return self.flatten().tolist(), self.shape

    @staticmethod
    def from_list(data_tuple: Tuple[List[float], Tuple[int, ...]]) -> 'Image':
        """Create Image instance from serialized data.

        Args:
            data_tuple: A tuple containing:
                - Flattened image data as a list
                - Desired shape of the image

        Returns:
            Image: New Image instance with the specified shape

        Example:
            .. code-block:: python

                # Reconstruct image from serialized data
                image = Image.from_list((data_list, shape))
        """
        data, shape = data_tuple
        # Ensure shape is a tuple of integers
        shape = tuple(map(int, shape))
        return Image(np.array(data).reshape(shape))

    def plot(self, ax: Optional[Any] = None) -> None:
        """Plot the image using matplotlib.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure.

        Note:
            This method automatically handles empty images and sets up
            appropriate display parameters for clean visualization.
        """
        # An empty Image (e.g. Image(None) -> np.array([]), shape (0,)) has no
        # data to show. The old `len(self.shape) == 0` only caught 0-d arrays,
        # so an empty image fell through to imshow() on an invalid (0,) shape.
        if self.size == 0:
            print("No image")
        else:
            import matplotlib.pyplot as plt
            if ax is None:
                plt.imshow(self)
                plt.gca().grid(False)
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                plt.show()
            else:
                # Hide axes ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(self)
                ax.grid(False)

