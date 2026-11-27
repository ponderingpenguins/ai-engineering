import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PointCloud(pd.DataFrame):    
    # This class variable tells Pandas the name of the attributes
    # that are to be ported over to derivative DataFrames.  There
    # is a method named `__finalize__` that grabs these attributes
    # and assigns them to newly created `PointCloud`
    _metadata = ['_xyz_vec', '_xyz_vecn']

    def __init__(self, data=None, *args, **kwargs):
        """Initialize PointCloud with data.
        
        Args:
            data: Point cloud data as numpy array or pandas DataFrame
            *args: Additional arguments for pandas DataFrame
            **kwargs: Additional keyword arguments
        """
        self._xyz_vec = kwargs.pop('xyzVec', None)
        self._xyz_vecn = kwargs.pop('xyzVecN', None)
        
        if data is not None and not isinstance(data, pd.DataFrame):
            # Convert numpy array to DataFrame
            if hasattr(data, 'shape') and len(data.shape) == 2 and data.shape[1] >= 3:
                # Assume it's a numpy array with x, y, z coordinates
                data = pd.DataFrame(data, columns=['x', 'y', 'z'])
            else:
                # Convert to DataFrame
                data = pd.DataFrame(data)
        
        super(PointCloud, self).__init__(data, *args, **kwargs)
        
        # Store original numpy array for compatibility
        if data is not None:
            if hasattr(data, 'values'):
                self._numpy_array = data.values
            elif hasattr(data, 'shape'):
                self._numpy_array = data
            else:
                self._numpy_array = np.array([])
        else:
            self._numpy_array = np.array([])

    def __array__(self, dtype=None):
        """Make PointCloud work as a numpy array."""
        return np.asarray(self._numpy_array, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support numpy ufuncs on PointCloud."""
        if method == '__call__':
            return ufunc(self._numpy_array, *inputs[1:], **kwargs)
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        """Support numpy array functions on PointCloud."""
        if func in (np.array_equal, np.array_equiv):
            return func(self._numpy_array, *args[1:], **kwargs)
        return NotImplemented

    def __array_wrap__(self, result):
        """Wrap numpy operations to return PointCloud objects."""
        if isinstance(result, np.ndarray):
            return PointCloud(result)
        return result

    def __class_getitem__(cls, item):
        """Support generic type hints."""
        return cls

    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `PointCloud` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `_xyz_vec` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finalize__` every time."""
        def _c(*args, **kwargs):
            return PointCloud(self._xyz_vec, self._xyz_vecn, *args, **kwargs).__finalize__(self)
        return _c

    def cluster(self, method=None):
        pass

    def plot_3d(self):
        fig = plt.figure()
        nice = fig.add_subplot(111, projection='3d')  # Updated to use add_subplot for compatibility

        nice.set_zlim3d(bottom=-5, top=5)
        nice.set_ylim(bottom=0, top=10)
        nice.set_xlim(left=-4, right=4)
        nice.set_xlabel('X Label')
        nice.set_ylabel('Y Label')
        nice.set_zlabel('Z Label')

        nice.scatter(self._xyz_vec[0], self._xyz_vec[1], self._xyz_vec[2], c='r', marker='o', s=2)
        plt.show()

    def plot(self):
        """Simple plot method for testing."""
        if hasattr(self, '_xyz_vec') and self._xyz_vec is not None:
            self.plot_3d()
        else:
            # Fallback to simple scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if len(self) > 0:
                ax.scatter(self['x'], self['y'], self['z'], c='r', marker='o', s=2)
            plt.show()

    def plot_xy(self):
        fig, axes = plt.subplots(1, 2)
        xyzVec = self._xyz_vec
        xyzVecN = self._xyz_vecn
        xyzVec = xyzVec[:, (np.abs(xyzVec[2]) < 1.5)]
        xyzVecN = xyzVecN[:, (np.abs(xyzVecN[2]) < 1.5)]
        axes[0].set_ylim(bottom=0, top=10)
        axes[0].set_ylabel('Range')
        axes[0].set_xlim(left=-4, right=4)
        axes[0].set_xlabel('Azimuth')
        axes[0].grid(visible =True)

        axes[1].set_ylim(bottom=0, top=10)
        axes[1].set_xlim(left=-4, right=4)
        axes[1].set_xlabel('Azimuth (m)')
        axes[1].grid(visible =True)
        axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=3)
        axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=3)

    def plot_xz(self):
        xyzVec = self._xyz_vec
        xyzVecN = self._xyz_vecn
    
        fig, axes = plt.subplots(1, 2)

        axes[0].set_ylim(bottom=-5, top=5)
        axes[0].set_ylabel('Elevation')
        axes[0].set_xlim(left=-4, right=4)
        axes[0].set_xlabel('Azimuth')
        axes[0].grid(visible =True)

        axes[1].set_ylim(bottom=-5, top=5)
        axes[1].set_xlim(left=-4, right=4)
        axes[1].set_xlabel('Azimuth')
        axes[1].grid(visible =True)


        axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=3)
        axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=3)

    def project(self):
        pass


# PointCloud implements numpy array interface through __array__ method

if __name__ == "__main__":
    pc = PointCloud({ "a" : [1,2,3,4], "b" : [3,4,5,6]})
    print(pc.describe())

