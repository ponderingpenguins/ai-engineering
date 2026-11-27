import numpy as np



def dp_reshape2LaneLVDS(rawData):
    # Convert 2 lane LVDS data to one matrix 
    rawData4 = np.reshape(rawData, (4, len(rawData)//4))
    rawDataI = np.reshape(rawData4[0:2, :], (-1, 1), order='F')
    rawDataQ = np.reshape(rawData4[2:4, :], (-1, 1), order='F')
    
    frameData = np.concatenate((rawDataI, rawDataQ), axis=1)
    
    return frameData

def dp_reshape4LaneLVDS(rawData):
    # Convert 4 lane LVDS data to one matrix 
    rawData8 = np.reshape(rawData, (8, len(rawData)//8))
    rawDataI = np.reshape(rawData8[0:4, :], (-1, 1), order='F')
    rawDataQ = np.reshape(rawData8[4:8, :], (-1, 1), order='F')
    
    frameData = np.concatenate((rawDataI, rawDataQ), axis=1)
    
    return frameData

def dp_generateFrameData(rawData):
    global Params
    global dataSet
    
    if Params['numLane'] == 2:
        # Convert 2 lane LVDS data to one matrix 
        frameData = dp_reshape2LaneLVDS(rawData)
    elif Params['numLane'] == 4:
        # Convert 4 lane LVDS data to one matrix 
        frameData = dp_reshape4LaneLVDS(rawData)
    else:
        print("%d LVDS lane is not supported " % Params['numLane'])
        return None
    
    # checking iqSwap setting
    if Params['adcDataParams']['iqSwap'] == 1:
        # Data is in ReIm format, convert to ImRe format to be used in radarCube 
        frameData[:, [0, 1]] = frameData[:, [1, 0]]
    
    # Convert data to complex: column 1 - Imag, 2 - Real
    frameCplx = frameData[:, 0] + 1j * frameData[:, 1]
    
    # initialize frameComplex. Must be complex — frameCplx below carries the
    # I+jQ samples, and a float32 array silently drops the imaginary (Q) half.
    frameComplex = np.zeros((Params['NChirp'], Params['NChan'], Params['NSample']), dtype=np.complex64)
    
    # Change Interleave data to non-interleave 
    if Params['chInterleave'] == 1:
        # non-interleave data
        temp = np.reshape(frameCplx, (Params['NSample'] * Params['NChan'], Params['NChirp'])).T
        for chirp in range(Params['NChirp']):
            frameComplex[chirp, :, :] = np.reshape(temp[chirp, :], (Params['NSample'], Params['NChan'])).T
    else:
        # interleave data
        temp = np.reshape(frameCplx, (Params['NSample'] * Params['NChan'], Params['NChirp'])).T
        for chirp in range(Params['NChirp']):
            frameComplex[chirp, :, :] = np.reshape(temp[chirp, :], (Params['NChan'], Params['NSample']))
    
    # Save raw data
    dataSet['rawFrameData'] = frameComplex

# You would need to define dp_reshape2LaneLVDS and dp_reshape4LaneLVDS functions
# as they are not provided in the original MATLAB code.
# Also, make sure to define the global variables Params and dataSet before calling the function.
