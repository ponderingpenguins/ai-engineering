import numpy as np
import matplotlib.pyplot as plt

class RangeCube(np.ndarray):
    """
        dimension 1 : number of ADC samples
        dimension 2 : number of RX
        dimension 3 : number of chirps
    """
    def __new__(cls, cube):
        assert cube is not None
        obj = np.asarray(cube).view(cls)
        return obj
    
    def plot_cube(self, ant_idx):
        plt.figure(figsize=(8, 4))
        plt.imshow(np.abs(self[:,ant_idx,:]), aspect='auto', cmap='viridis')
        plt.title('Frequency Content of ADC Samples (FFT Result RX' + str(ant_idx+1)+ ')')
        plt.xlabel('Sample')
        plt.ylabel('Chirp Index')
        plt.colorbar(label='Magnitude')
        plt.show()

    def plot_rangeinfo(self, chirp, antenna, loga=False):
        fft_sample = self[chirp,antenna,:]
        
        plt.figure(figsize=(8, 4))
        
        # Determine full scale(maximum possible amplitude) by bit depth
        bit_depth = 16
        full_scale = 2 ** bit_depth

        # dbFS function
        to_dbFS = lambda x: 20 * np.log10(x / full_scale)

        data = np.absolute(fft_sample)
        if loga:
            data = np.log(data)
        plt.plot(to_dbFS(data))


        plt.title('FFT Result of ADC Samples (chirp ' + str(chirp) + ')')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()
