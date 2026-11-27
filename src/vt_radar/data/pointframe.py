import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vt_radar.mmwave.dsp as dsp

class PointFrame(pd.DataFrame):    
    # This class variable tells Pandas the name of the attributes
    # that are to be ported over to derivative DataFrames.  There
    # is a method named `__finalize__` that grabs these attributes
    # and assigns them to newly created `PointCloud`
    _metadata = ['_xyz_vec', '_xyz_vecn']

    def __init__(self, *args, **kwargs):
        self._xyz_vec = kwargs.pop('xyzVec', None)
        self._xyz_vecn = kwargs.pop('xyzVecN', None)
        super(PointFrame, self).__init__(*args, **kwargs)
          
    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `PointFrame` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `_xyz_vec` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finalize__` every time."""
        def _c(*args, **kwargs):
            return PointFrame(self._xyz_vec, self._xyz_vecn, *args, **kwargs).__finalize__(self)
        return PointFrame

    def cluster(self, method=None):
        pass

    @staticmethod
    def doit(detObj2DRaw):
        # Create a DataFrame and unwrap the location tuple
        detObj2D_df = PointFrame.from_records(detObj2DRaw.tolist(),
                                              columns=['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'])
        print("A", detObj2D_df.describe())
        # Extract and create separate columns for 'x', 'y', 'z' from the 'location' column
        detObj2D_df[['x', 'y', 'z']] = pd.DataFrame(detObj2D_df['location'].tolist(), index=detObj2D_df.index)

        detObj2D_df.reindex()
        print("B", detObj2D_df.describe())
        # Drop the original 'location' column
        detObj2D_df = detObj2D_df.drop(columns=['location'])

        return detObj2D_df
    
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

    def prune(self, det_matrix, num_doppler_bins, num_range_bins, range_resolution):
        detObj2DRaw = dsp.prune_to_peaks(self, det_matrix, num_doppler_bins, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, 
                                                   det_matrix, 
                                                num_doppler_bins)
        
        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D,
                                        SNRThresholds2, 
                                        peakValThresholds2, 
                                        num_range_bins, # max_range
                                        0.5, # min_range
                                        range_resolution)
        return detObj2D
    
if __name__ == "__main__":
    pc = PointFrame({ "a" : [1,2,3,4], "b" : [3,4,5,6]})
    print(pc.describe())

