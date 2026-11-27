import struct
import sys
# import serial
import binascii
import time
import numpy as np
import math
from typing import Dict, Tuple, Optional, Union

import os
import datetime

# Local File Imports
from .parseTLVs import *
from ..tlv_constants import *

# Constants
POINT_CLOUD_COLUMNS: int = 7  # X, Y, Z, Doppler, SNR, Noise, TrackIndex
NO_TRACK_INDEX: int = 255     # Value indicating no track assigned

def parseStandardFrame(frameData: bytes, config, verbose: bool = False) -> 'RawRadarData':
    headerStruct = 'Q8I'
    frameHeaderLen = struct.calcsize(headerStruct)
    tlvHeaderLength = 8

    outputDict: Dict[str, Union[int, np.ndarray, Dict[str, Union[int, float, np.ndarray]]]] = {}
    outputDict['error'] = 0

    try:
        # Read in frame Header
        magic: int
        version: int
        totalPacketLen: int
        platform: int
        frameNum: int
        timeCPUCycles: int
        numDetectedObj: int
        numTLVs: int
        subFrameNum: int
        magic, version, totalPacketLen, platform, frameNum, timeCPUCycles, numDetectedObj, numTLVs, subFrameNum = struct.unpack(headerStruct, frameData[:frameHeaderLen])
    except:
        if verbose:
            logger.error('Error: Could not read frame header')
        outputDict['error'] = 1
        logger.exception("Exception", exc_info=True)
    # Move frameData ptr to start of 1st TLV    
    frameData = frameData[frameHeaderLen:]

    # Save frame number to output
    outputDict['frameNum'] = frameNum

    # Initialize the point cloud struct since it is modified by multiple TLV's
    # Each point has the following: X, Y, Z, Doppler, SNR, Noise, Track index
    outputDict['pointCloud'] = np.zeros((numDetectedObj, POINT_CLOUD_COLUMNS), np.float64)
    # Initialize the track indexes to a value which indicates no track
    outputDict['pointCloud'][:, 6] = NO_TRACK_INDEX
    # Find and parse all TLV's
    for i in range(numTLVs):
        try:
            tlvType: int
            tlvLength: int
            tlvType, tlvLength = tlvHeaderDecode(frameData[:tlvHeaderLength])
            frameData = frameData[tlvHeaderLength:]
        except:
            if verbose:
                logger.error('TLV Header Parsing Failure')
            outputDict['error'] = 2

        # Use `or` (short-circuit): when the TLV header decode above failed,
        # error is 2 and tlvLength is unbound, so the right operand must not be
        # evaluated. Bitwise `|` evaluated both and crashed with UnboundLocalError
        # on any truncated/corrupt TLV header.
        if (outputDict['error'] == 2) or (tlvLength > len(frameData)):
            if verbose:
                logger.error("Ignored frame due to parsing error")
            break

        # Detected Points
        if (tlvType == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parsePointCloudTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Range Profile
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_PROFILE):
            outputDict['rangeProfile'] = parseRangeProfileTLV(frameData[:tlvLength])
        # Range Profile
        elif (tlvType == MMWDEMO_OUTPUT_EXT_MSG_RANGE_PROFILE_MAJOR):
            outputDict['rangeProfileMajor'] = parseRangeProfileTLV(frameData[:tlvLength])
        # Range Profile
        elif (tlvType == MMWDEMO_OUTPUT_EXT_MSG_RANGE_PROFILE_MINOR):
            outputDict['rangeProfileMinor'] = parseRangeProfileTLV(frameData[:tlvLength])
        # Noise Profile
        elif (tlvType == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE):
            logger.error("MMWDEMO_OUTPUT_MSG_NOISE_PROFILE")
            pass
        # Static Azimuth Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP):
            logger.error("MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP")
            pass
        # Range Doppler Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP):
            outputDict["rangeDopplerHeatmap"] = parseRangeDopplerHeatmap(
                frameData[:tlvLength], 
                num_doppler_bins=config.sensor().num_doppler_bins, 
                num_range_bins=config.sensor().num_range_bins
                )
        # Performance Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_STATS):
            outputDict['stats'] = parseStatsTLV(frameData[:tlvLength])
        # Side Info (SNR and Noise)
        elif (tlvType == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO):
            outputDict['pointCloud'] = parseSideInfoTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
         # Azimuth Elevation Static Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP):
            logger.error("MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP")
            pass
        # Temperature Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS):
            outputDict['temperatureStats'] = parseTemperatureStatsTLV(frameData[:tlvLength])
        # Spherical Points
        elif (tlvType == MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parseSphericalPointCloudTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Target 3D
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST or tlvType == MMWDEMO_OUTPUT_EXT_MSG_TARGET_LIST):
            outputDict['numDetectedTracks'], outputDict['trackData'] = parseTrackTLV(frameData[:tlvLength], tlvLength)
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT):
            outputDict['numDetectedHeights'], outputDict['heightData'] = parseTrackHeightTLV(frameData[:tlvLength], tlvLength)
         # Target index
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX or tlvType ==  MMWDEMO_OUTPUT_EXT_MSG_TARGET_INDEX):
            outputDict['trackIndexes'] = parseTargetIndexTLV(frameData[:tlvLength], tlvLength)
         # Capon Compressed Spherical Coordinates
        elif (tlvType == MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'], outputDict["pointMeta"] = parseCompressedSphericalPointCloudTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Presence Indication (is there an object in the bounding box)
        elif (tlvType == MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION):
            occStateMachStruct = 'I' # Single uint32_t which holds 32 booleans
            occStateMachLength = struct.calcsize(occStateMachStruct)
            outputDict["presenceIndication"] = struct.unpack(occStateMachStruct, frameData[:occStateMachLength])
        # Occupancy State Machine
        elif (tlvType == MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE):
            outputDict['occupancy'] = parseOccStateMachTLV(frameData[:tlvLength])
        elif (tlvType == MMWDEMO_OUTPUT_MSG_VITALSIGNS):
            outputDict['vitals'] = parseVitalSignsTLV(frameData[:tlvLength], tlvLength)
        elif(tlvType == MMWDEMO_OUTPUT_EXT_MSG_DETECTED_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parsePointCloudExtTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        elif (tlvType == MMWDEMO_OUTPUT_MSG_GESTURE_FEATURES_6843):
            outputDict['features'] = parseGestureFeaturesTLV(frameData[:tlvLength])
        elif (tlvType == MMWDEMO_OUTPUT_MSG_GESTURE_OUTPUT_PROB_6843):
            outputDict['gestureNeuralNetProb'] = parseGestureProbTLV6843(frameData[:tlvLength])
        elif (tlvType == MMWDEMO_OUTPUT_MSG_GESTURE_FEATURES_6432): # 6432 features output 350
            pass
        elif (tlvType == MMWDEMO_OUTPUT_MSG_GESTURE_CLASSIFIER_6432):
            outputDict['gesture'] = parseGestureClassifierTLV6432(frameData[:tlvLength])
        # Performance Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_EXT_STATS):
            logger.error("Performance statistics")
        # Presence Detection in each zone
        elif(tlvType == MMWDEMO_OUTPUT_EXT_MSG_ENHANCED_PRESENCE_INDICATION):
            outputDict['enhancedPresenceDet'] = parseEnhancedPresenceInfoTLV(frameData[:tlvLength], tlvLength)
        # Probabilities output by the classifier
        elif(tlvType == MMWDEMO_OUTPUT_EXT_MSG_CLASSIFIER_INFO):
            outputDict['classifierOutput'] = parseClassifierTLV(frameData[:tlvLength], tlvLength)
        # Raw data from uDoppler extraction around targets
        elif(tlvType == MMWDEMO_OUTPUT_EXT_MSG_MICRO_DOPPLER_RAW_DATA):
            logger.error("MMWDEMO_OUTPUT_EXT_MSG_MICRO_DOPPLER_RAW_DATA")
        # uDoppler features from each target
        elif(tlvType == MMWDEMO_OUTPUT_EXT_MSG_MICRO_DOPPLER_FEATURES):
            logger.error("MMWDEMO_OUTPUT_EXT_MSG_MICRO_DOPPLER_FEATURES")
        else:
            if verbose:
                logger.error("Warning: invalid TLV type: %d" % (tlvType))

        # logger.info ("Frame Data after tlv parse: ", frameData[:10])
        # Move to next TLV
        frameData = frameData[tlvLength:]
        # logger.info ("Frame Data at end of TLV: ", frameData[:10])
    
    from ..data.raw_radar_data import RawRadarData
    return RawRadarData.from_dict(outputDict, frameData)


# Decode TLV Header
def tlvHeaderDecode(data: bytes) -> Tuple[int, int]:
    tlvType: int
    tlvLength: int
    tlvType, tlvLength = struct.unpack('2I', data)
    return tlvType, tlvLength

