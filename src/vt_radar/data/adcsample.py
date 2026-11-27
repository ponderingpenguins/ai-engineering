import numpy as np
import matplotlib.pyplot as plt

class ADCSample(np.ndarray):
    def __new__(cls, sample, chirp_idx=None, ant_idx=None):
        obj = np.asarray(sample).view(cls)
        obj.chirp_idx = chirp_idx
        obj.ant_idx = ant_idx
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.chirp_idx = getattr(obj, 'chirp_idx', None)
        self.ant_idx = getattr(obj, 'ant_idx', None)

    def plot(self, sample_time=None):
        """
            Input is a complex array
        """

        N = len(self)
        T = sample_time / N
        f = 1/(T / 1e9)
        xf = np.linspace(0.0, f, N)

        #to_dbFS = lambda x: 20 * np.log10(x / full_scale)


        # Plotting real part
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 1, 1)
        plt.plot(xf, np.real(self), color='b')
        plt.plot(xf, np.imag(self), color='r')
        plt.title('Time domain ADC chirp %d ant %d' % (self.chirp_idx, self.ant_idx))
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        # Plotting imaginary part
    #    plt.subplot(3, 1, 2)
      #  plt.title('Imaginary Part')
      #  plt.xlabel('Sample Index')
      #  plt.ylabel('Amplitude')

        # Plotting magnitude part
        plt.subplot(2, 1, 2)
        plt.plot(np.abs(self))
        plt.title('Magnitude Part')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.show()

        