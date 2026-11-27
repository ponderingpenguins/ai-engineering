import numpy as np
import vt_radar.mmwave.dsp as dsp
from vt_radar.config.ti_config import RadarSS
import matplotlib.pyplot as plt
from vt_radar.data.adcsample import ADCSample
from vt_radar.data.rangecube import RangeCube
from vt_radar.data.rangedoppler import RangeDoppler

from vt_radar.mmwave.dsp.utils import Window
from vt_radar.mmwave.dsp.doppler_processing import separate_tx

#TODO(JB) this is nicer and can be used with caching
class RadarFrame(object):
    def __init__(self, raw_cube, sensor: RadarSS):
        assert raw_cube is not None
        self._raw_cube = raw_cube
        self._sensor = sensor
        #Properties
        angle_res = 1

        angle_range = 120
        origin_at_bottom_center = True
        use_float32 = False
        self.__range_azimuth_dirty = True
        self.virt_ant = sensor.num_rx_antennas * sensor.num_tx_antennas #radar_cube.shape[1]
        self.angle_bins = (angle_range * 2) // angle_res + 1
        print(self._sensor.num_range_bins)
        self.__range_azimuth = np.zeros((sensor.num_range_bins, self.angle_bins),
                                        dtype=np.complex64 if use_float32 else np.complex_)
        self.__beam_weights = np.zeros((self.virt_ant, sensor.num_range_bins),
                                       dtype=np.complex64 if use_float32 else np.complex_)
        print(self.__range_azimuth.shape)
        self.__range_doppler = None
        self.__range_cube = None

        print("virt_ant", self.virt_ant)
        self.num_vec, self.steering_vec = dsp.gen_steering_vec(angle_range,
                                                               angle_res,
                                                               self.virt_ant)

        self.__flip_ra = origin_at_bottom_center

    def doppler_processing(self, clutter_removal_enabled=True):
        # automatic detection, not yet verified
        interleaved = self._radar_cube.shape[1] <= 4
        det_matrix, aoa_input = dsp.doppler_processing(self._radar_cube, 
                                               num_tx_antennas=self._sensor.num_tx_antennas, 
                                               interleaved=interleaved,
                                               clutter_removal_enabled=clutter_removal_enabled, 
                                               window_type_2d=Window.HAMMING)
        # the result is the averages range-doppler and the range-doppler for every virtual antenna
        return RangeDoppler(det_matrix, self._sensor), aoa_input

    @property
    def range_azimuth_capon(self):
        """Get the radar data as a range azimuth cube,
        beamformed using the Capon beamformer.
        This property is computed lazily on first access.
        """
        if not self.__range_azimuth_dirty:
            r = self.__range_azimuth
        else:
            self.__aoa_capon_process()
            r = self.__range_azimuth

        if self.__flip_ra:
            return np.flipud(np.fliplr(r))
        else:
            return r
        
    @property
    def range_cube(self) -> RangeCube:
        # the input here doesn't matter if it is virtual or RX since FFT is done over 
        # the last axis
        print(self._raw_cube.shape)
        range_cube = dsp.range_processing(self._raw_cube, window_type_1d=Window.BLACKMAN)
        # nr chirps is now axis 2
        self.__range_cube = np.swapaxes(range_cube, 0, 2)
        self._radar_cube = range_cube
#        self.__range_cube = range_cube
        return RangeCube(range_cube)
        
    @property
    def range_doppler(self):
        range_doppler = dsp.doppler_processing(self._raw_cube, interleaved=False)
        #clutter_removal_enabled=False,
        #               interleaved=True,
        #               window_type_2d=None,
        #               accumulate=True
        self.__range_doppler = range_doppler
        return self.__range_doppler
    
    def sample(self, chirp_idx, ant_idx: int) -> ADCSample:
        return ADCSample(self._raw_cube[chirp_idx,ant_idx,:], chirp_idx, ant_idx)
    
    @property
    def detection_matrix_size(self):
        # sizeof(uint16_t)
        return  2 #* num_range_bins * num_doppler_bins

    def __aoa_capon_process(self):
        radar_cube = self.raw_cube
        print(radar_cube.shape)
        for jj in range(self._sensor.num_range_bins):
            self.__range_azimuth[jj,:], self.__beam_weights[:,jj] = dsp.aoa_capon(radar_cube[jj],
                                                                      self.steering_vec)

        self.__range_azimuth_dirty = False

    def compute_range_azimuth(self, radar_raw=None, method='capon'):
        """Beamform raw radar datacube
        Use this method to set the the raw radar data cube and 
        perform beamforming. The beamformed data cube will be returned.
        
        Currently only the capon method is implemented.

        This is a convenience method that is equivalent to:
        ```
        # rf = RadarFrame(...)
        rf.raw_cube = radar_raw
        beamformed_datacube = rf.range_azimuth_capon
        ```
        """
        if radar_raw is not None:
            self.raw_cube = radar_raw

        if method == 'capon':
            return self.range_azimuth_capon
        else:
            raise NotImplementedError
