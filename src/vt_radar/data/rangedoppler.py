import numpy as np
import matplotlib.pyplot as plt

from vt_radar.config.ti_config import RadarSS

def convert_power_to_db(x):
    """Convert the power values of the range-doppler heatmap to decibel
    source: https://github.com/m6c7l/pymmw/blob/master/source/lib/helper.py
    """
    x = x / (1 << 9)
    return x * 6

class RangeDoppler(np.ndarray):
    def __new__(cls, sample, sensor=None):
        obj = np.asarray(sample).view(cls)
        obj._sensor = sensor
        return obj

    def __init__(self, sample, sensor: RadarSS):
        self._sensor = sensor

    def __array_finalize__(self, obj):
        # numpy calls this (not __init__) for every array derived from a
        # RangeDoppler — slice, .copy(), .view(), arithmetic (e.g.
        # convert_power_to_db does self / 512 * 6). Without propagating _sensor the
        # derived array lacks the attribute entirely, so the plot methods that read
        # self._sensor raise AttributeError. Inherit it, defaulting to None.
        if obj is None:
            return
        self._sensor = getattr(obj, "_sensor", None)

    def _plot_doppler_range_profile(self, ax):
        """Plot the range profile at zero doppler"""
        range_bins = self._sensor.range_bins

        range_dopppler_data = convert_power_to_db(self)

        doppler_0_profile = np.flip(range_dopppler_data[:,0])
        ax.plot(range_bins, doppler_0_profile)
        ax.set_title("Range profile for 0 Doppler")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Power (dB)")
        ax.set_xlim([0, max(range_bins)])

    def _plot_heatmap(self, ax, plot_colorbar=True):
        """
            returns ..
        """
        doppler_bins = self._sensor.doppler_bins
        range_bins = self._sensor.range_bins
       
        det_matrix_vis = np.fft.fftshift(self, axes=1)
        det_matrix_vis = np.flipud(det_matrix_vis)

        heatmap = ax.imshow(
            det_matrix_vis / det_matrix_vis.max(), 
            cmap='viridis', # jet is weird
            interpolation='bicubic', 
            extent=(min(doppler_bins), max(doppler_bins), 0, max(range_bins))
        )

        extent = heatmap.get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))

        ax.set_title("RangeDoppler Heatmap")
        ax.set_xlabel("Doppler (m/s)")
        ax.set_ylabel("Range (m)")
        
        if plot_colorbar:
            # Add a colorbar to the heatmap
            cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04, ax=ax, 
                                ticks=[np.min(self), np.max(self)])
            cbar.ax.set_yticklabels([np.min(self), np.max(self)])
            cbar.set_label("Power (in which unit ?")  # Add a label to the colorbar
        
        return heatmap

    def plot_heatmap(self):
        fig, ax = plt.subplots(1, 1)
        self._plot_heatmap(ax, fig)

    def plot_range_profile(self):
        fig, ax = plt.subplots(1, 1)
        self._plot_doppler_range_profile(ax)
 
    def plot_all(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 8))
        ax = ax.ravel()

        self._plot_heatmap(ax[0])
        self._plot_doppler_range_profile(ax[1])
        fig.tight_layout()

