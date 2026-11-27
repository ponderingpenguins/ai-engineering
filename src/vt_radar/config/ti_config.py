from loguru import logger
from typing import Any, List, Optional
import numpy as np
import tabulate
import json
from dataclasses import dataclass, fields as dataclass_fields
from ..tlv_constants import *
from .base import Cfg
from .radar_subsystem import (
    RadarSS, ChannelCfg, AdcCfg, AdcBufCfg, AntGeometry, AntPhaseRot,
    ChirpCfg, ProfileCfg, FrameCfg, ClutterRemoval, CalibDcRangeSig,
    CompRangeBiasAndRxChanPhase, LvdsStreamCfg, antenna_enabled,
    AdvFrameCfg, SubFrameCfg, AoaFovCfg, MeasureRangeBiasAndRxChanPhase, CalibData
)

NUM_HEART_RATES_FOR_MEDIAN = 10
NUM_FRAMES_PER_VITALS_PACKET = 15
MAX_TX = 3
MAX_RX = 4


class Board(object):
    """Physical settings belonging to the hardware board"""


class CFAR(object):
    # This is the category CFAR DPU
    def __init__(self):
        pass


class GTRACK(object):
    """
    The DPU for GTRACK, the configuration parameters can be subdivided in:
    - scenery
    - gating (physical contraints)
    - allocation (new tracks)
    - state (tracking state transition)
    - acceleration
    """

    def __init__(self):
        # the bounding box coordinates in which the tracker is allowed to operate
        self.boundaryBox = None
        # the bounding box in which targets are becoming static for longer periods of time
        self.staticBoundaryBox = None
        # the bounding box is the origin of tracks
        self.presenceBoundaryBox = None


class BoundaryBox(Cfg):
    """
    Boundary box configuration for defining spatial limits.

    This configuration defines the 3D spatial boundaries for tracking and detection.
    It sets the limits in X, Y, and Z dimensions to constrain the search space
    for target detection and tracking algorithms.

    Parameters:
        xMin (float): Minimum X coordinate in meters
        xMax (float): Maximum X coordinate in meters
        yMin (float): Minimum Y coordinate in meters
        yMax (float): Maximum Y coordinate in meters
        zMin (float): Minimum Z coordinate in meters
        zMax (float): Maximum Z coordinate in meters

    Valid Values:
        - xMin, xMax: -100.0 to 100.0 meters
        - yMin, yMax: -100.0 to 100.0 meters
        - zMin, zMax: 0.0 to 20.0 meters (typical ceiling height)

    Constraints:
        - xMin < xMax
        - yMin < yMax
        - zMin < zMax

    Usage:
        - Defines detection and tracking boundaries
        - Improves processing efficiency by limiting search space
        - Critical for people counting applications
        - Must match physical room dimensions

    Example:
        boundaryBox -5.0 5.0 -5.0 5.0 0.0 3.0  # 10x10x3 meter room
    """

    def __init__(self) -> None:
        self.xMin: float = 0.0
        self.xMax: float = 0.0
        self.yMin: float = 0.0
        self.yMax: float = 0.0
        self.zMin: float = 0.0
        self.zMax: float = 0.0

    def parse(self, args: List[str]) -> None:
        if len(args) < 7:
            raise ValueError("BoundaryBox requires at least 7 arguments")
        self.xMin = float(args[1])
        self.xMax = float(args[2])
        self.yMin = float(args[3])
        self.yMax = float(args[4])
        self.zMin = float(args[5])
        self.zMax = float(args[6])

    def describe(self) -> None:
        table = [
            ["X Min", self.xMin],
            ["X Max", self.xMax],
            ["Y Min", self.yMin],
            ["Y Max", self.yMax],
            ["Z Min", self.zMin],
            ["Z Max", self.zMax]
        ]
        print(tabulate.tabulate(table))


class PresenceBoundaryBox(BoundaryBox):
    pass


class StaticBoundaryBox(BoundaryBox):
    pass


class FovCfg(Cfg):
    """
    Field of View (FOV) configuration for radar detection.

    This command defines the sensor azimuth and elevation FOVs. The FOV values
    define the total angular extent observed at one side of the sensor. A value
    of θ configures the corresponding FOV to lie in the closed interval [−θ, θ].
    Targets outside of this angular FOV interval are not detected.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        azimuthFoV (float): Azimuth field of view in degrees
        elevationFoV (float): Elevation field of view in degrees

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - azimuthFoV: 1.0 to 180.0 degrees
        - elevationFoV: 1.0 to 180.0 degrees

    Usage:
        - Defines angular coverage for target detection
        - Limits processing to specified angular range
        - Improves processing efficiency
        - Critical for accurate target tracking

    Example:
        fovCfg -1 70.0 70.0  # 70-degree azimuth and elevation FOV
    """

    def __init__(self) -> None:
        self.subFrameIdx: int = -1
        self.azimuthFoV: float = 0.0
        self.elevationFoV: float = 0.0

    def parse(self, args: List[str]) -> None:
        """
        Parse FOV configuration parameters from command line arguments.

        Args:
            args (list): Command line arguments containing FOV parameters
        """
        if len(args) < 4:
            raise ValueError("fovCfg requires at least 4 arguments")
        self.subFrameIdx = int(args[1])
        self.azimuthFoV = float(args[2])
        self.elevationFoV = float(args[3])


class DynamicRangeAngleCfg(Cfg):
    """
    Dynamic range-angle configuration for Capon beamforming.

    For dynamic scenes, the range-angle heatmap estimation uses Capon beamforming
    in the 2D range-azimuth domain for 3D wall-mount chain and 3D range-azimuth-elevation
    domain for 3D ceil-mount chain. This configuration optimizes beamforming
    parameters for moving target detection.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        angleSearchStep (float): Angle search step in degrees
        rangeAngleDiagonalLoading (float): Diagonal loading factor
        rangeAngleEstMethod (int): Range-angle estimation method (0-1)
        dopplerEstMethod (int): Doppler estimation method (0-1)

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - angleSearchStep: 0.1 to 10.0 degrees
        - rangeAngleDiagonalLoading: 0.001 to 1.0
        - rangeAngleEstMethod: 0-1
        - dopplerEstMethod: 0-1

    Usage:
        - Optimizes beamforming for dynamic scenes
        - Configures Capon beamforming parameters
        - Improves moving target detection
        - Critical for 3D tracking applications

    Example:
        dynamicRangeAngleCfg -1 0.75 0.0010 1 0  # 0.75° step, 0.001 loading
    """

    def parse(self, args: List[str]) -> None:
        self.subFrameIdx = int(args[1])
        self.angleSearchStep = float(args[2])
        self.rangeAngleDiagonalLoading = float(args[3])
        self.rangeAngleEstMethod = int(args[4])
        self.dopplerEstMethod = int(args[5])


class DynamicRACfarCfg(Cfg):
    """
    Dynamic Range-Angle CFAR configuration.

    This configuration is part of the detection chain for dynamic scenes.
    It configures the Constant False Alarm Rate algorithm for range-angle
    processing in moving target scenarios.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        rangeWinSize (int): Range window size for CFAR processing
        rangeStepSize (int): Range step size for CFAR processing
        angleWinSize (int): Angle window size for CFAR processing
        angleStepSize (int): Angle step size for CFAR processing
        dopplerWinSize (int): Doppler window size for CFAR processing
        dopplerStepSize (int): Doppler step size for CFAR processing
        cfarThreshold (float): CFAR detection threshold

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - rangeWinSize: 4, 8, 12, 16, 20, 24, 28, 32
        - rangeStepSize: 1-8
        - angleWinSize: 4, 8, 12, 16, 20, 24, 28, 32
        - angleStepSize: 1-8
        - dopplerWinSize: 4, 8, 12, 16, 20, 24, 28, 32
        - dopplerStepSize: 1-8
        - cfarThreshold: 0.1 to 10.0

    Usage:
        - Optimizes CFAR for dynamic scenes
        - Configures window sizes and step sizes
        - Sets detection threshold
        - Critical for moving target detection

    Example:
        dynamicRACfarCfg -1 8 2 8 2 8 2 1.5  # 8x8x8 windows, step 2, threshold 1.5
    """

    def parse(self, args: List[str]) -> None:
        self.subFrameIdx = int(args[1])
        self.cfarDiscardLeftRange = int(args[2])
        self.cfarDiscardRightRange = int(args[3])
        self.cfarDiscardLeftAngle = int(args[4])
        self.cfarDiscardRightAngle = int(args[5])
        self.refWinSizeRange = int(args[6])
        self.refWinSizeAngle = int(args[7])
        self.guardWinSizeRange = int(args[8])
        self.guardWinSizeAngle = int(args[9])
        self.rangeThre = float(args[10])
        self.angleThre = float(args[11])
        self.sidelobeThre = float(args[12])
        self.enable2ndPass = int(args[13])
        self.dynamicFlag = int(args[14])


class StaticRACfarCfg(Cfg):
    """
    Static Range-Angle CFAR configuration.

    This configuration is part of the detection chain for static scenes.
    It configures the Constant False Alarm Rate algorithm for range-angle
    processing in stationary target scenarios.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        rangeWinSize (int): Range window size for CFAR processing
        rangeStepSize (int): Range step size for CFAR processing
        angleWinSize (int): Angle window size for CFAR processing
        angleStepSize (int): Angle step size for CFAR processing
        cfarThreshold (float): CFAR detection threshold

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - rangeWinSize: 4, 8, 12, 16, 20, 24, 28, 32
        - rangeStepSize: 1-8
        - angleWinSize: 4, 8, 12, 16, 20, 24, 28, 32
        - angleStepSize: 1-8
        - cfarThreshold: 0.1 to 10.0

    Usage:
        - Optimizes CFAR for static scenes
        - Configures window sizes and step sizes
        - Sets detection threshold
        - Critical for stationary target detection

    Example:
        staticRACfarCfg -1 8 2 8 2 1.5  # 8x8 windows, step 2, threshold 1.5
    """

    def parse(self, args: List[str]) -> None:
        self.subFrameIdx = int(args[1])
        self.cfarDiscardLeftRange = int(args[2])
        self.cfarDiscardRightRange = int(args[3])
        self.cfarDiscardLeftAngle = int(args[4])
        self.cfarDiscardRightAngle = int(args[5])
        self.refWinSizeRange = int(args[6])
        self.refWinSizeAngle = int(args[7])
        self.guardWinSizeRange = int(args[8])
        self.guardWinSizeAngle = int(args[9])
        self.rangeThre = float(args[10])
        self.angleThre = float(args[11])
        self.sidelobeThre = float(args[12])
        self.enable2ndPass = int(args[13])
        self.dynamicFlag = int(args[14])


class StateParam(Cfg):
    """
    State parameter configuration for tracking state machine.

    This configuration defines the thresholds for state transitions in the
    tracking algorithm. It controls how targets move between different states
    (detection, active, free, static, exit, sleep) based on detection history
    and tracking confidence.

    Parameters:
        det2actThre (int): Detection to active threshold (number of detections)
        det2freeThre (int): Detection to free threshold (number of detections)
        active2freeThre (int): Active to free threshold (number of detections)
        static2freeThre (int): Static to free threshold (number of detections)
        exit2freeThre (int): Exit to free threshold (number of detections)
        sleep2freeThre (int): Sleep to free threshold (number of detections)

    Valid Values:
        - det2actThre: 1-10 (typically 2-3)
        - det2freeThre: 1-20 (typically 5-10)
        - active2freeThre: 1-20 (typically 5-10)
        - static2freeThre: 1-20 (typically 5-10)
        - exit2freeThre: 1-20 (typically 5-10)
        - sleep2freeThre: 1-20 (typically 5-10)

    Usage:
        - Controls tracking state machine behavior
        - Balances detection sensitivity vs. false alarms
        - Critical for stable target tracking
        - Must be tuned for specific applications

    Example:
        stateParam 2 5 5 5 5 5  # Conservative thresholds for stable tracking
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 7:
            raise ValueError("stateParam requires at least 7 arguments")
        self.det2actThre = int(args[1])
        self.det2freeThre = int(args[2])
        self.active2freeThre = int(args[3])
        self.static2freeThre = int(args[4])
        self.exit2freeThre = int(args[5])
        self.sleep2freeThre = int(args[6])

    def describe(self) -> None:
        table = [
            ["Detection to Active Threshold", self.det2actThre],
            ["Detection to Free Threshold", self.det2freeThre],
            ["Active to Free Threshold", self.active2freeThre],
            ["Static to Free Threshold", self.static2freeThre],
            ["Exit to Free Threshold", self.exit2freeThre],
            ["Sleep to Free Threshold", self.sleep2freeThre]
        ]
        print(tabulate.tabulate(table))


class BpmCfg(Cfg):
    """
    BPM (Beats Per Minute) configuration for vital signs monitoring.

    This configuration enables vital signs detection and monitoring for heart rate
    and breathing rate. It's used in healthcare applications and presence detection
    to monitor human vital signs using radar technology.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        enabled (int): Enable vital signs monitoring (0=disabled, 1=enabled)
        heartRate (int): Heart rate in beats per minute
        breathRate (int): Breathing rate in breaths per minute

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - enabled: 0-1
        - heartRate: 40-200 BPM (typical human range)
        - breathRate: 8-40 BPM (typical human range)

    Usage:
        - Enables vital signs monitoring
        - Critical for healthcare applications
        - Used in presence detection
        - Requires stable radar setup and target positioning

    Example:
        bpmCfg -1 1 70 16  # Enable, 70 BPM heart rate, 16 BPM breathing
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 5:
            raise ValueError("bpmCfg requires at least 5 arguments")
        self.subFrameIdx = int(args[1])
        self.enabled = int(args[2])
        self.heartRate = int(args[3])
        self.breathRate = int(args[4])

    def describe(self) -> None:
        table = [
            ["Sub Frame Index", self.subFrameIdx],
            ["Enabled", self.enabled],
            ["Heart Rate", self.heartRate],
            ["Breath Rate", self.breathRate]
        ]
        print(tabulate.tabulate(table))


class GatingParam(Cfg):
    """
        = Gain: Gain is defined as the amount the track gating limits specified in the limit parameters below can
            grow to create the gating function. Each of the dimensions X (width), Y (depth), Z (height) and Velocity
            is multiplied by the gain to generate a gating function for the track. This value is set based on expected
            tracking errors, expected target maneuver and uncertainties in the detection layer. We have empirically
            chose a value of 3 based on “three sigma rule” in statistics.
        = Limit-Width, Limit-Depth, Limit-Height, Limit-Velocity: These values determine the limits of the gating
            function and are set based on physical dimensions and agility of the targets. Setting these too small will
            result in allocating multiple tracks for the single object, setting these too large will cause allocating single
            track for multiple objects. To set these limit values, the physical dimensions of the object that are to be
            tracked can be used as a good starting point and then the values further optimized based on test results.
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 6:
            raise ValueError("gatingParam requires at least 6 arguments")
        self.gain = float(args[1])
        self.limitWidth = float(args[2])  # m
        self.limitDepth = float(args[3])  # m
        self.limitHeight = float(args[4])  # m
        self.limitVelocity = float(args[5])  # m/s


class CfarCfg(Cfg):
    """
    Constant False Alarm Rate (CFAR) configuration parameters.

    CFAR is a detection algorithm used to maintain a constant false alarm rate
    by adaptively setting the detection threshold based on the noise level in
    the surrounding environment.

    Parameters:
        subFrameIdx (int): Sub-frame index for CFAR processing
        procDirection (int): Processing direction (0=range, 1=doppler, 2=angle)
        mode (int): CFAR mode (0=CA-CFAR, 1=OS-CFAR, 2=GO-CFAR, 3=SO-CFAR)
        noiseWin (int): Noise window size for CFAR processing
        guardLen (int): Guard length to avoid target self-cancellation
        divShift (int): Division shift for threshold calculation
        cyclic (int): Cyclic processing flag (0=non-cyclic, 1=cyclic)
        thresholdScale (int): Threshold scaling factor (typically 0-15)
        peakGrouping (int): Peak grouping enable flag (0=disabled, 1=enabled)
    """

    def parse(self, args: List[str]) -> None:
        """
        Parse CFAR configuration parameters from command line arguments.

        Args:
            args (list): Command line arguments containing CFAR parameters
        """
        if len(args) < 10:
            raise ValueError("cfarCfg requires at least 10 arguments")

        self.subFrameIdx = int(args[1])
        self.procDirection = int(args[2])
        self.mode = int(args[3])
        self.noiseWin = int(args[4])
        self.guardLen = int(args[5])
        self.divShift = int(args[6])
        self.cyclic = int(args[7])
        self.thresholdScale = int(args[8])
        self.peakGrouping = int(args[9])

        # Range checks based on TI documentation.
        # subFrameIdx == -1 is the TI convention for "apply to all subframes"
        # (widely used in real radar configs generated by the Visualizer).
        if self.subFrameIdx < -1:
            raise ValueError("subFrameIdx must be >= -1 (-1 = all subframes)")
        if self.procDirection not in [0, 1, 2]:
            raise ValueError("procDirection must be 0 (range), 1 (doppler), or 2 (angle)")
        if self.mode not in [0, 1, 2, 3]:
            raise ValueError("mode must be 0 (CA-CFAR), 1 (OS-CFAR), 2 (GO-CFAR), or 3 (SO-CFAR)")
        if self.noiseWin < 1 or self.noiseWin > 32:
            raise ValueError("noiseWin must be between 1 and 32")
        if self.guardLen < 0 or self.guardLen > 16:
            raise ValueError("guardLen must be between 0 and 16")
        if self.divShift < 0 or self.divShift > 15:
            raise ValueError("divShift must be between 0 and 15")
        if self.cyclic not in [0, 1]:
            raise ValueError("cyclic must be 0 (non-cyclic) or 1 (cyclic)")
        if self.thresholdScale < 0 or self.thresholdScale > 15:
            raise ValueError("thresholdScale must be between 0 and 15")
        if self.peakGrouping not in [0, 1]:
            raise ValueError("peakGrouping must be 0 (disabled) or 1 (enabled)")

    def describe(self) -> None:
        """Print a formatted description of the CFAR configuration."""
        table = [
            ["Sub-frame Index", self.subFrameIdx],
            ["Processing Direction", ["Range", "Doppler", "Angle"][self.procDirection]],
            ["CFAR Mode", ["CA-CFAR", "OS-CFAR", "GO-CFAR", "SO-CFAR"][self.mode]],
            ["Noise Window", self.noiseWin],
            ["Guard Length", self.guardLen],
            ["Division Shift", self.divShift],
            ["Cyclic", "Yes" if self.cyclic else "No"],
            ["Threshold Scale", self.thresholdScale],
            ["Peak Grouping", "Enabled" if self.peakGrouping else "Disabled"]
        ]
        print(tabulate.tabulate(table))


class CfarFovCfg(Cfg):
    """
    CFAR Field of View (FOV) configuration parameters.

    This configuration is used to filter out detected points outside the
    specified limits in the range direction or doppler direction.

    Parameters:
        subFrameIdx (int): Sub-frame index for CFAR FOV processing
        procDirection (int): Processing direction (0=range, 1=doppler, 2=angle)
        min (float): Minimum value for FOV filtering
        max (float): Maximum value for FOV filtering
    """

    def parse(self, args: List[str]) -> None:
        """
        Parse CFAR FOV configuration parameters from command line arguments.

        Args:
            args (list): Command line arguments containing CFAR FOV parameters
        """
        if len(args) < 5:
            raise ValueError("cfarFovCfg requires at least 5 arguments")

        self.subFrameIdx = int(args[1])
        self.procDirection = int(args[2])
        self.min = float(args[3])
        self.max = float(args[4])

        # Range checks based on TI documentation.
        # subFrameIdx == -1 is the TI convention for "apply to all subframes".
        if self.subFrameIdx < -1:
            raise ValueError("subFrameIdx must be >= -1 (-1 = all subframes)")
        if self.procDirection not in [0, 1, 2]:
            raise ValueError("procDirection must be 0 (range), 1 (doppler), or 2 (angle)")
        if self.min >= self.max:
            raise ValueError("min value must be less than max value")

    def describe(self) -> None:
        """Print a formatted description of the CFAR FOV configuration."""
        table = [
            ["Sub-frame Index", self.subFrameIdx],
            ["Processing Direction", ["Range", "Doppler", "Angle"][self.procDirection]],
            ["Minimum Value", self.min],
            ["Maximum Value", self.max]
        ]
        print(tabulate.tabulate(table))


class MaxAcceleration(Cfg):
    """
    Maximum acceleration configuration for tracking.

    This configuration defines the maximum acceleration limits for target tracking
    in X, Y, and Z dimensions. These limits are used to constrain the tracking
    algorithm and prevent unrealistic target motion predictions.

    Parameters:
        maxX (float): Maximum acceleration in X direction (m/s²)
        maxY (float): Maximum acceleration in Y direction (m/s²)
        maxZ (float): Maximum acceleration in Z direction (m/s²)

    Valid Values:
        - maxX: 0.1 to 50.0 m/s² (typical human: 2-5 m/s²)
        - maxY: 0.1 to 50.0 m/s² (typical human: 2-5 m/s²)
        - maxZ: 0.1 to 20.0 m/s² (typical human: 1-3 m/s²)

    Usage:
        - Constrains tracking algorithm predictions
        - Prevents unrealistic target motion
        - Improves tracking stability
        - Must be tuned for target type (human, vehicle, etc.)

    Example:
        maxAcceleration 3.0 3.0 2.0  # 3 m/s² horizontal, 2 m/s² vertical
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 4:
            raise ValueError("maxAcceleration requires at least 4 arguments")
        self.maxX = float(args[1])
        self.maxY = float(args[2])
        self.maxZ = float(args[3])

    def describe(self) -> None:
        table = [
            ["Max X Acceleration", self.maxX],
            ["Max Y Acceleration", self.maxY],
            ["Max Z Acceleration", self.maxZ]
        ]
        print(tabulate.tabulate(table))


class MultiObjBeamForming(Cfg):
    """
    Multi Object Beamforming configuration.

    This feature allows radar to separate reflections from multiple
    objects originating from the same range/Doppler detection.
    The procedure searches for the second peak after locating the
    highest peak in Azimuth FFT. If the second peak is greater than
    the specified threshold, the second object with the same range
    /Doppler is appended to the list of detected objects. The
    threshold is proportional to the height of the highest peak.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        enabled (int): Enable multi-object beamforming (0=disabled, 1=enabled)
        peakThreshold (float): Peak detection threshold (proportional to highest peak)
        minPeakSeparation (float): Minimum separation between peaks in degrees

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - enabled: 0-1
        - peakThreshold: 0.1 to 1.0 (proportional to highest peak)
        - minPeakSeparation: 1.0 to 45.0 degrees

    Usage:
        - Enables separation of multiple objects at same range/Doppler
        - Improves detection accuracy in crowded scenarios
        - Critical for people counting applications
        - Requires sufficient angular resolution

    Example:
        multiObjBeamForming -1 1 0.5 5.0  # Enable, 0.5 threshold, 5° separation
    """

    def parse(self, args: List[str]) -> None:
        # Reads args[1..3]; guard like the sibling cfg classes (CfarFovCfg,
        # MaxAcceleration, FineMotionCfg) instead of IndexError'ing on a short line.
        if len(args) < 4:
            raise ValueError("multiObjBeamForming requires at least 4 arguments")
        self.subFrameIdx = int(args[1])  # should be set to -1
        self.enabled = int(args[2])
        self.threshold = float(args[3])


class FineMotionCfg(Cfg):
    """
    Fine motion configuration for micro-movement detection.

    This configuration enables fine motion processing for detecting
    very small movements such as breathing, heartbeats, or subtle
    gestures. It's particularly useful for vital signs monitoring
    and presence detection applications.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        fineMotionProcEnabled (int): Enable fine motion processing (0=disabled, 1=enabled)
        fineMotionObservationTime (float): Observation time window in seconds

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - fineMotionProcEnabled: 0-1
        - fineMotionObservationTime: 1.0 to 60.0 seconds

    Usage:
        - Enables micro-movement detection
        - Critical for vital signs monitoring
        - Used in presence detection applications
        - Requires stable radar setup

    Example:
        fineMotionCfg -1 1 10.0  # Enable, 10-second observation window
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 3:
            raise ValueError("fineMotionCfg requires at least 3 arguments")
        self.subFrameIdx = int(args[1])
        self.fineMotionProcEnabled = int(args[2])
        self.fineMotionObservationTime = float(args[3]) if len(args) > 3 else 10.0
        self.fineMotionProcCycle = int(args[4]) if len(args) > 4 else 0
        self.fineMotionDopplerThrIdx = int(args[5]) if len(args) > 5 else 0


class Dynamic2DAngleCfg(Cfg):
    """
    Dynamic 2D Angle configuration for elevation estimation.

    When the range-angle heatmap estimation method 1 is used (i.e., rangeAngleEstMethod = 1),
    for each point detected in the range-azimuth domain, a Capon beamforming algorithm is
    applied to generate the elevation spectrum, which will be used to estimate the elevation
    angle of the detected point.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        angleSearchStep (float): Angle search step in degrees
        elevationThreshold (float): Elevation detection threshold
        maxElevationAngle (float): Maximum elevation angle in degrees

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - angleSearchStep: 0.1 to 10.0 degrees
        - elevationThreshold: 0.1 to 10.0
        - maxElevationAngle: 1.0 to 90.0 degrees

    Usage:
        - Enables elevation angle estimation for 3D tracking
        - Uses Capon beamforming for elevation spectrum
        - Critical for 3D people counting applications
        - Requires rangeAngleEstMethod = 1

    Example:
        dynamic2DAngleCfg -1 0.5 1.0 60.0  # 0.5° step, 1.0 threshold, 60° max
    """

    def parse(self, args: List[str]) -> None:
        # Reads args[1..9]; guard like the sibling cfg classes (CfarFovCfg,
        # MaxAcceleration, ...) instead of IndexError'ing on a truncated line.
        if len(args) < 10:
            raise ValueError("dynamic2DAngleCfg requires at least 10 arguments")
        self.subFrameIdx = int(args[1])
        self.elevSearchStep = float(args[2])
        self.angleDiagonalLoading = float(args[3])
        self.maxNpeak2Search = int(args[4])
        self.peakExpSamples = int(args[5])
        self.elevOnly = int(args[6])
        self.sidelobeThre = float(args[7])
        self.peakExpRelThre = float(args[8])
        self.peakExpSNRThre = float(args[9])


class SensorPosition(Cfg):
    """
    Sensor position configuration for radar mounting.

    This configuration defines the physical position and orientation
    of the radar sensor. It includes height, azimuth tilt, and elevation
    tilt parameters for accurate target localization.

    Parameters:
        sensorHeight (float): Sensor height above ground in meters
        azTilt (float): Azimuth tilt angle in degrees
        elevTilt (float): Elevation tilt angle in degrees

    Valid Values:
        - sensorHeight: 0.1 to 10.0 meters
        - azTilt: -180.0 to 180.0 degrees
        - elevTilt: -90.0 to 90.0 degrees

    Usage:
        - Defines sensor mounting position
        - Critical for accurate target localization
        - Used in coordinate transformations
        - Affects tracking and detection accuracy

    Example:
        sensorPosition 2.0 0 15  # 2m height, no azimuth tilt, 15° elevation tilt
    """

    def describe(self) -> None:
        table = [
            ["sensor_height", self.sensorHeight],
            ["az_tilt", self.azTilt],
            ["elev_tilt", self.elevTilt],
        ]
        print(tabulate.tabulate(table))

    def parse(self, args: List[str]) -> None:
        self.sensorHeight = float(args[1])
        self.azTilt = float(args[2])
        self.elevTilt = float(args[3])


class StaticRangeAngleCfg(Cfg):
    """
    Static Range-Angle configuration for Bartlett beamforming.

    The range-angle heatmap estimation step uses Bartlett beamforming in the
    range-azimuth-elevation domain for static scenes. This configuration
    optimizes beamforming parameters for stationary target detection.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        angleSearchStep (float): Angle search step in degrees
        rangeAngleDiagonalLoading (float): Diagonal loading factor
        rangeAngleEstMethod (int): Range-angle estimation method (0-1)

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - angleSearchStep: 0.1 to 10.0 degrees
        - rangeAngleDiagonalLoading: 0.001 to 1.0
        - rangeAngleEstMethod: 0-1

    Usage:
        - Optimizes beamforming for static scenes
        - Uses Bartlett beamforming algorithm
        - Improves stationary target detection
        - Critical for static monitoring applications

    Example:
        staticRangeAngleCfg -1 0.75 0.0010 0  # 0.75° step, 0.001 loading, method 0
    """

    def parse(self, args: List[str]) -> None:
        self.subFrameIdx = int(args[1])
        self.staticProcEnabled = int(args[2])
        self.staticAzimStepDeciFactor = int(args[3])
        self.staticElevStepDeciFactor = int(args[4])


class GuiMonitorCfg:
    """
    GUI Monitor configuration for visualization settings.

    This configuration controls what information is displayed in the GUI interface.
    It enables or disables various visualization components and monitoring features
    for radar data analysis and system debugging.

    Parameters:
        detectedObjects (int): Display detected objects (0=disabled, 1=enabled)
        logMagRange (int): Display log magnitude range plot (0=disabled, 1=enabled)
        noiseProfile (int): Display noise profile (0=disabled, 1=enabled)
        rangeAzimuthElevationHeatMap (int): Display range-azimuth-elevation heatmap (0=disabled, 1=enabled)
        rangeDopplerHeatMap (int): Display range-Doppler heatmap (0=disabled, 1=enabled)
        statsInfo (int): Display statistics information (0=disabled, 1=enabled)

    Valid Values:
        - detectedObjects: 0-1
        - logMagRange: 0-1
        - noiseProfile: 0-1
        - rangeAzimuthElevationHeatMap: 0-1
        - rangeDopplerHeatMap: 0-1
        - statsInfo: 0-1

    Usage:
        - Controls GUI visualization components
        - Enables debugging and analysis features
        - Improves user experience
        - Can be adjusted for different use cases

    Example:
        guiMonitor 1 1 1 1 1 1  # Enable all visualization components
    """

    def __getitem__(self, __name: str) -> Any:
        # Attributes are stored without an underscore prefix (parse()/describe() use
        # self.detectedObjects, ...). The "_" + name form read a missing key (KeyError)
        # and, via __setitem__, silently wrote to a phantom "_"-attr the reads never saw.
        return self.__dict__[__name]

    def __setitem__(self, __name: str, __value) -> Any:
        self.__dict__[__name] = __value

    def parse(self, args: List[str]) -> None:
        if len(args) < 8:
            raise ValueError("guiMonitor requires at least 8 arguments")
        self.detectedObjects = int(args[1])
        self.logMagRange = int(args[2])
        self.noiseProfile = int(args[3])
        self.rangeAzimuthElevationHeatMap = int(args[4])
        self.rangeDopplerHeatMap = int(args[5])
        self.statsInfo = int(args[6])

    def describe(self) -> None:
        table = [
            ["detectedObjects", str(self.detectedObjects)],
            ["logMagRange ", str(self.logMagRange)],
            ["noiseProfile ", str(self.noiseProfile)],
            ["rangeAzimuthElevationHeatMap", str(self.rangeAzimuthElevationHeatMap)],
            ["rangeDopplerHeatMap ", self.rangeDopplerHeatMap],
            ["statsInfo ", str(self.statsInfo)],
        ]
        print(tabulate.tabulate(table))


class PmicCfg(Cfg):
    """
    Power Management IC (PMIC) configuration.

    This configuration controls power management settings for the radar system.
    It enables or disables various power management features and sets operating
    modes for optimal power consumption and performance.

    Parameters:
        enable (int): Enable PMIC (0=disabled, 1=enabled)
        mode (int): PMIC operating mode (0=low power, 1=normal, 2=high performance)

    Valid Values:
        - enable: 0-1
        - mode: 0-2

    Usage:
        - Controls radar system power consumption
        - Enables power management features
        - Sets operating mode for performance vs. power trade-off
        - Critical for battery-powered applications

    Example:
        pmicCfg 1 1  # Enable PMIC in normal mode
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 3:
            raise ValueError("pmicCfg requires at least 3 arguments")
        self.enable = int(args[1])
        self.mode = int(args[2])

    def describe(self) -> None:
        table = [
            ["Enable", self.enable],
            ["Mode", self.mode]
        ]
        print(tabulate.tabulate(table))


class VitalSignCfg(Cfg):
    """
    Vital signs configuration for health monitoring.

    This configuration enables vital signs detection and monitoring using radar
    technology. It's used in healthcare applications to monitor heart rate,
    breathing rate, and other vital signs without physical contact.

    Parameters:
        rangeBin (int): Range bin index for vital signs detection
        timeout (int): Timeout for vital signs detection in seconds

    Valid Values:
        - rangeBin: 0-255 (depends on range resolution)
        - timeout: 1-300 seconds

    Usage:
        - Enables non-contact vital signs monitoring
        - Critical for healthcare applications
        - Used in presence detection and health monitoring
        - Requires stable target positioning

    Example:
        vitalsign 10 30  # Range bin 10, 30-second timeout
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 3:
            raise ValueError("vitalsign requires at least 3 arguments")
        self.rangeBin = int(args[1])
        self.timeout = int(args[2])

    def describe(self) -> None:
        table = [
            ["Range Bin", self.rangeBin],
            ["Timeout", self.timeout]
        ]
        print(tabulate.tabulate(table))


class VSRangeIdxCfg(Cfg):
    """
    Vital Signs Range Index configuration.

    This configuration defines the range index parameters for vital signs detection.
    It specifies which range bins to use for monitoring vital signs and sets
    the range limits for effective detection.

    Parameters:
        rangeIdx (int): Range index for vital signs monitoring
        rangeMin (float): Minimum range for vital signs detection in meters
        rangeMax (float): Maximum range for vital signs detection in meters

    Valid Values:
        - rangeIdx: 0-255 (depends on range resolution)
        - rangeMin: 0.1 to 10.0 meters
        - rangeMax: 0.1 to 10.0 meters

    Constraints:
        - rangeMin < rangeMax
        - rangeIdx should correspond to the target range

    Usage:
        - Defines range limits for vital signs detection
        - Optimizes detection for specific target distances
        - Critical for accurate vital signs monitoring
        - Must be tuned for target positioning

    Example:
        VSRangeIdxCfg 5 0.5 2.0  # Range index 5, 0.5-2.0m range
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 3:
            raise ValueError("VSRangeIdxCfg requires at least 3 arguments")
        self.subFrameIdx = int(args[1])
        self.rangeIdx = int(args[2])

    def describe(self) -> None:
        table = [
            ["Sub Frame Index", self.subFrameIdx],
            ["Range Index", self.rangeIdx]
        ]
        print(tabulate.tabulate(table))


class CQRxSatMonitor(Cfg):
    """
    RX Saturation Quality Monitor configuration.

    This configuration enables monitoring of RX saturation levels for quality control.
    It helps detect when the receiver is saturated and provides quality metrics
    for system diagnostics and optimization.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        enabled (int): Enable RX saturation monitoring (0=disabled, 1=enabled)
        threshold (float): Saturation threshold level
        windowSize (int): Window size for saturation detection

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - enabled: 0-1
        - threshold: 0.1 to 1.0 (normalized level)
        - windowSize: 1-100 (number of samples)

    Usage:
        - Monitors receiver saturation levels
        - Provides quality control metrics
        - Helps optimize radar performance
        - Critical for system diagnostics

    Example:
        CQRxSatMonitor -1 1 0.8 10  # Enable, 0.8 threshold, 10-sample window
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 6:
            raise ValueError("CQRxSatMonitor requires at least 6 arguments")
        self.subFrameIdx = int(args[1])
        self.enable = int(args[2])
        self.threshold = int(args[3])
        self.numSamples = int(args[4])
        self.timeout = int(args[5])

    def describe(self) -> None:
        table = [
            ["Sub Frame Index", self.subFrameIdx],
            ["Enable", self.enable],
            ["Threshold", self.threshold],
            ["Num Samples", self.numSamples],
            ["Timeout", self.timeout]
        ]
        print(tabulate.tabulate(table))


class CQSigImgMonitor(Cfg):
    """
    Signal Image Quality Monitor configuration.

    This configuration enables monitoring of signal image quality for system
    diagnostics. It provides quality metrics for radar signal processing
    and helps identify issues with signal integrity.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        enabled (int): Enable signal image monitoring (0=disabled, 1=enabled)
        threshold (float): Quality threshold level
        windowSize (int): Window size for quality assessment

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - enabled: 0-1
        - threshold: 0.1 to 1.0 (normalized quality level)
        - windowSize: 1-100 (number of samples)

    Usage:
        - Monitors signal image quality
        - Provides system diagnostics
        - Helps optimize signal processing
        - Critical for quality control

    Example:
        CQSigImgMonitor -1 1 0.7 15  # Enable, 0.7 threshold, 15-sample window
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 4:
            raise ValueError("CQSigImgMonitor requires at least 4 arguments")
        self.subFrameIdx = int(args[1])
        self.threshold = int(args[2])
        self.numSamples = int(args[3])

    def describe(self) -> None:
        table = [
            ["Sub Frame Index", self.subFrameIdx],
            ["Threshold", self.threshold],
            ["Num Samples", self.numSamples]
        ]
        print(tabulate.tabulate(table))


@dataclass
class AdvFrameCfgData:
    """Typed storage for advFrameCfg sub-fields."""
    numSubFrames: int = 0
    subFrameIdx: int = 0
    chirpStartIdx: int = 0
    chirpEndIdx: int = 0
    numLoops: int = 0


@dataclass
class RadarProfile:
    """Typed radar profile replacing the legacy Dict[str, Any].

    Fields always initialized with defaults come first;
    fields populated during parseCfg() are Optional with None default.
    Backward-compatible dict-style access (__getitem__, __setitem__,
    __contains__, get()) is provided for notebook/legacy code.
    """
    # --- Always-initialized defaults ---
    startFreq: float = 60.25
    sensorHeight: float = 3.0
    maxRange: float = 10.0
    az_tilt: float = 0.0
    elev_tilt: float = 0.0
    enabled: float = 0.0

    # --- Set during parseCfg (profileCfg) ---
    id: Optional[int] = None
    idle: Optional[float] = None
    adcStart: Optional[float] = None
    rampEnd: Optional[float] = None
    txOutPower: Optional[float] = None
    txPhaseShifer: Optional[float] = None  # typo preserved — matches TI SDK naming
    slope: Optional[float] = None
    txStartTime: Optional[float] = None
    samples: Optional[float] = None
    sampleRate: Optional[float] = None

    # --- Set during parseCfg (various commands) ---
    maxTracks: Optional[int] = None
    boundaryBox: Optional[List[float]] = None
    advFrameCfg: Optional['AdvFrameCfgData'] = None
    rxSaturation: Optional[int] = None
    sigImgBand: Optional[int] = None
    lowPower: Optional[int] = None
    extendedMaxVelocity: Optional[int] = None
    dfeDataOutputMode: Optional[int] = None
    maxDistance: Optional[float] = None
    minPoints: Optional[float] = None

    # --- Computed after parsing ---
    rangeIdxToMeters: Optional[float] = None
    maxVelocity: Optional[float] = None

    # --- x432 family (dead code path) ---
    xOffset: Optional[float] = None
    yOffset: Optional[float] = None

    def __getitem__(self, key: str) -> Any:
        """Dict-style read access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-style write access for backward compatibility."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Dict-style 'in' check. Returns False when field is None (not set)."""
        return hasattr(self, key) and getattr(self, key) is not None

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style .get() with None-means-absent semantics."""
        value = getattr(self, key, None)
        if value is None:
            return default
        return value

    def __repr__(self) -> str:
        """Dict-like repr for describe() output compatibility."""
        d = {f.name: getattr(self, f.name) for f in dataclass_fields(self)
             if getattr(self, f.name) is not None}
        return repr(d)


class ConfigFile(object):
    def __init__(self, device: str = "IWR6843AOP", show_warning: bool = False) -> None:
        self._device = device
        self._show_warning = show_warning
        self.frameTime = 50
        self.graphFin = 1
        self.hGraphFin = 1
        self.threeD = 1
        self.lastFramePoints = np.zeros((5, 1))
        self.plotTargets = 1
        self.frameNum = 0
        self.profile: RadarProfile = RadarProfile()
        self.chirpComnCfg = {
            'DigOutputSampRate': 23,
            'DigOutputBitsSel': 0,
            'DfeFirSel': 0,
            'NumOfAdcSamples': 128,
            'ChirpTxMimoPatSel': 4,
            'ChirpRampEndTime': 36.1,
            'ChirpRxHpfSel': 1}
        self.chirpTimingCfg = {
            'ChirpIdleTime': 8,
            'ChirpAdcSkipSamples': 24,
            'ChirpTxStartTime': 0,
            'ChirpRfFreqSlope': 47.95,
            'ChirpRfFreqStart': 60}
        self.rangeRes = 0
        self.rangeAxisVals = np.zeros(int(self.chirpComnCfg['NumOfAdcSamples'] / 2))
        self.sensorHeight = 1.5
        self.numFrameAvg = 10
        self.configSent = 0
        self.previousFirstZ = -1
        self.yzFlip = 0
        # timer to reset fall detected message

        self.fallResetTimerOn = 0
        self.fallThresh = -0.22

        self.trackColorMap = None
        self.vitalsPatientData = []

        # class based data storage
        self._chirps = []
        self._frames = []
        self._profiles = []
        self._guimonitor = None
        self._clutter_removal = None
        self._cfarCfgs = []
        self._cfarFovCfgs = []
        self._fov_cfg = None
        self.gtrack = GTRACK()
        self._sensor = RadarSS()

    def __str__(self):
        if self._clutter_removal:
            return str(self._sensor.num_doppler_bins) + "x" + str(self._sensor.num_range_bins) + \
                " clutter_removal=" + str(self._clutter_removal.enabled())
        else:
            return str(self._sensor.num_doppler_bins) + "x" + str(self._sensor.num_range_bins)

    def describe(self) -> None:
        for p in self._profiles:
            p.describe()
        for f in self._frames:
            f.describe()
        for c in self._cfarFovCfgs:
            c.describe()
        if self._guimonitor:
            self._guimonitor.describe()
        # print("Frame periocidy", self.profile[""])
        print("Transmit antennas ", int(self._sensor.num_tx_antennas))
        print(self.profile)

    def to_json(self) -> dict:
        return self.to_dict()

    def parseCfg(self, fname: str) -> None:
        with open(fname, 'r') as cfg_file:
            self.cfg = cfg_file.readlines()
        counter = 0
        chirpCount = 0
        for line in self.cfg:
            args = line.split()
            if (len(args) > 0):
                # cfarCfg
                if (args[0] == 'cfarCfg'):
                    cfar_cfg = CfarCfg()
                    cfar_cfg.parse(args)
                    self._cfarCfgs.append(cfar_cfg)
                elif (args[0] == "adcCfg"):
                    adc_cfg = AdcCfg()
                    adc_cfg.parse(args)
                    self._adc_cfg = adc_cfg
                elif (args[0] == "adcbufCfg"):
                    adc_buf_cfg = AdcBufCfg()
                    adc_buf_cfg.parse(args)
                    self._sensor._adc_buf_cfg = adc_buf_cfg
                elif (args[0] == 'antGeometry0'):
                    ant_geometry = AntGeometry()
                    ant_geometry.parse(args)
                    self._ant_geometry0 = ant_geometry
                elif (args[0] == 'antGeometry1'):
                    ant_geometry = AntGeometry()
                    ant_geometry.parse(args)
                    self._ant_geometry1 = ant_geometry
                elif (args[0] == 'antPhaseRot'):
                    ant_phase_rot = AntPhaseRot()
                    ant_phase_rot.parse(args)
                    self._ant_phase_rot = ant_phase_rot
                elif (args[0] == 'bpmCfg'):
                    bpm_cfg = BpmCfg()
                    bpm_cfg.parse(args)
                    self._bpm_cfg = bpm_cfg
                elif (args[0] == 'calibdata'):
                    logger.error("calibdata")
                elif (args[0] == 'calibDcRangeSig'):
                    calib = CalibDcRangeSig()
                    calib.parse(args)
                    self._calib_dc_range_sig = calib
                elif (args[0] == 'clutterRemoval'):
                    cr = ClutterRemoval()
                    cr.parse(args)
                    self._clutter_removal = cr
                elif (args[0] == 'dynamic2DAngleCfg'):
                    dynamic2DAngleCfg = Dynamic2DAngleCfg()
                    dynamic2DAngleCfg.parse(args)
                    self._dynamic2DAngleCfg = dynamic2DAngleCfg
                elif (args[0] == "dynamicRACfarCfg"):
                    dynamic_racfar_cfg = DynamicRACfarCfg()
                    dynamic_racfar_cfg.parse(args)
                    self._dynamic_racfar_cfg = dynamic_racfar_cfg
                elif (args[0] == 'fineMotionCfg'):
                    fmc = FineMotionCfg()
                    fmc.parse(args)
                    self._fineMotionCfg = fmc
                elif (args[0] == "staticRACfarCfg"):
                    static_racfar_cfg = StaticRACfarCfg()
                    static_racfar_cfg.parse(args)
                    self._static_racfar_cfg = static_racfar_cfg
                elif (args[0] == "multiObjBeamForming"):
                    moobf = MultiObjBeamForming()
                    moobf.parse(args)
                    self._multiObjBeamForming = moobf
                # trackingCfg
                elif (args[0] == 'trackingCfg'):
                    if (len(args) < 5):
                        print("Error: trackingCfg had fewer arguments than expected")
                        continue
                    # logger.error("trackingConfig")
                    self.profile.maxTracks = int(args[4])
                    # self.trackColorMap = get_trackColors(self.profile.maxTracks)
                    for m in range(self.profile.maxTracks):
                        # Add track gui object
                        #                        mesh = gl.GLLinePlotItem()
                        #                        mesh.setVisible(False)
                        #                        self.pcplot.addItem(mesh)
                        #                        self.ellipsoids.append(mesh)
                        #                         #add track coordinate string
                        #                        text = GLTextItem()
                        #                        text.setGLViewWidget(self.pcplot)
                        #                        text.setVisible(False)
                        #                        self.pcplot.addItem(text)
                        #                        self.coordStr.append(text)
                        pass
                    # If we only support 1 patient, hide the other patient window
                    if (self.profile.maxTracks == 1):
                        # print("self.vitals[1]['pane'].setVisible(False)")
                        pass
                    # Initialize Vitals output dictionaries for each potential patient
                    for i in range(min(self.profile.maxTracks, MAX_VITALS_PATIENTS)):
                        # Initialize
                        patientDict = {}
                        patientDict['id'] = i
                        patientDict['rangeBin'] = 0
                        patientDict['breathDeviation'] = 0
                        patientDict['heartRate'] = []
                        patientDict['breathRate'] = 0
                        patientDict['heartWaveform'] = []
                        patientDict['breathWaveform'] = []
                        self.vitalsPatientData.append(patientDict)
                elif (args[0] == 'dynamicRangeAngleCfg'):
                    drac = DynamicRangeAngleCfg()
                    drac.parse(args)
                    self._dynamicRangeAngleCfg = drac
                elif (args[0] == 'allocationParam'):
                    pass
                    # logger.error("allocationParam")
                elif (args[0] == 'staticRangeAngleCfg'):
                    staticRangeAngleCfg = StaticRangeAngleCfg()
                    staticRangeAngleCfg.parse(args)
                    self._staticRangeAngleCfg = staticRangeAngleCfg
                elif (args[0] == 'compRangeBiasAndRxChanPhase'):
                    c = CompRangeBiasAndRxChanPhase()
                    c.parse(args)
                    self._comp = c
                elif (args[0] == 'gatingParam'):
                    g = GatingParam()
                    g.parse(args)
                    self.gatingParam = g

                elif (args[0] == 'SceneryParam' or args[0] == 'boundaryBox'):
                    if (len(args) < 7):
                        logger.error("Error: SceneryParam/boundaryBox had fewer arguments than expected")
                        continue
                    self.boundaryLine = counter
                    leftX = float(args[1])
                    rightX = float(args[2])
                    nearY = float(args[3])
                    farY = float(args[4])
                    bottomZ = float(args[5])
                    topZ = float(args[6])
                    self.profile.boundaryBox = [leftX, rightX, nearY, farY, bottomZ, topZ]
                    # self.addBoundBox('trackerBounds', leftX, rightX, nearY, farY, bottomZ, topZ)
                    boundary_box = BoundaryBox()
                    boundary_box.xMin = leftX
                    boundary_box.xMax = rightX
                    boundary_box.yMin = nearY
                    boundary_box.yMax = farY
                    boundary_box.zMin = bottomZ
                    boundary_box.zMax = topZ
                    self.gtrack.boundaryBox = boundary_box
                elif (args[0] == 'LvdsStreamCfg'):
                    lvds = LvdsStreamCfg()
                    lvds.parse(args)
                    self._lvdsStreamCfg = lvds
                elif (args[0] == 'staticBoundaryBox'):
                    self.staticLine = counter
                    # this is part of the tracker
                elif (args[0] == 'presenceBoundaryBox'):
                    # Guard arg count like the sibling boundaryBox handler above:
                    # a short line here would IndexError and abort the whole cfg
                    # parse, whereas boundaryBox logs and continues.
                    if (len(args) < 7):
                        logger.error("Error: presenceBoundaryBox had fewer arguments than expected")
                        continue
                    leftX = float(args[1])
                    rightX = float(args[2])
                    nearY = float(args[3])
                    farY = float(args[4])
                    bottomZ = float(args[5])
                    topZ = float(args[6])

                    boundary_box = PresenceBoundaryBox()
                    boundary_box.xMin = leftX
                    boundary_box.xMax = rightX
                    boundary_box.yMin = nearY
                    boundary_box.yMax = farY
                    boundary_box.zMin = bottomZ
                    boundary_box.zMax = topZ
                    self.gtrack.presenceBoundaryBox = boundary_box
                elif (args[0] == 'profileCfg'):
                    # ProfileCfg.parse(args) below reads args[1..14], so the full
                    # 15-token line is required. The old `< 12` guard let 12-14
                    # token lines through and IndexError'd inside parse().
                    if (len(args) < 15):
                        logger.error("Error: profileCfg had fewer arguments than expected")
                        continue
                    self.profile.id = int(args[1])  # 0-3
                    self.profile.startFreq = float(args[2])
                    self.profile.idle = float(args[3])
                    self.profile.adcStart = float(args[4])
                    self.profile.rampEnd = float(args[5])
                    self.profile.txOutPower = float(args[6])
                    self.profile.txPhaseShifer = float(args[7])
                    self.profile.slope = float(args[8])
                    self.profile.txStartTime = float(args[9])
                    self.profile.samples = float(args[10])
                    self.profile.sampleRate = float(args[11])
                    # TODO: we need this since there can be more profiles
                    p = ProfileCfg()
                    p.parse(args)
                    self._profiles.append(p)
                    self._sensor.add_profile(p)

                elif (args[0] == 'advFrameCfg'):
                    # Advanced frame configuration
                    if len(args) < 6:
                        logger.error("Error: advFrameCfg had fewer arguments than expected")
                        continue
                    self.profile.advFrameCfg = AdvFrameCfgData(
                        numSubFrames=int(args[1]),
                        subFrameIdx=int(args[2]),
                        chirpStartIdx=int(args[3]),
                        chirpEndIdx=int(args[4]),
                        numLoops=int(args[5]),
                    )
                elif (args[0] == 'subFrameCfg'):
                    # Sub-frame configuration
                    if len(args) < 11:
                        logger.error("Error: subFrameCfg had fewer arguments than expected")
                        continue
                    sub_frame_cfg = {
                        "subFrameIdx": int(args[1]),
                        "chirpStartIdx": int(args[2]),
                        "chirpEndIdx": int(args[3]),
                        "numLoops": int(args[4]),
                        "framePeriodicity": float(args[5]),
                        "triggerSelect": int(args[6]),
                        "frameTriggerDelay": float(args[7]),
                        "numFrames": int(args[8])
                    }
                    if not hasattr(self, '_subFrameCfgs'):
                        self._subFrameCfgs = []
                    self._subFrameCfgs.append(sub_frame_cfg)
                elif (args[0] == 'frameCfg'):
                    f = FrameCfg()
                    if not f.parse(args):
                        continue
                    self._frames.append(f)
                    self._sensor.add_frame(f)
                elif (args[0] == 'chirpCfg'):
                    # note that this is for each antenna
                    # ChirpCfg.parse reads args[1..8], so 9 tokens are required.
                    # Guard like every other multi-arg command here; a short line
                    # would otherwise IndexError inside parse() and abort the
                    # entire cfg parse.
                    if (len(args) < 9):
                        logger.error("Error: chirpCfg had fewer arguments than expected")
                        continue
                    chirpCount += 1
                    c = ChirpCfg()
                    c.parse(args)
                    self._sensor.add_chirp(c)
                elif (args[0] == 'analogMonitor'):
                    # Controls the enable/disable of the various monitoring features
                    # supported in the demos
                    self.profile.rxSaturation = int(args[1])
                    self.profile.sigImgBand = int(args[2])
                elif (args[0] == 'lowPower'):
                    # args[1] is don't care (0)
                    self.profile.lowPower = int(args[2])
                elif (args[0] == 'extendedMaxVelocity'):
                    # mandatory
                    self.profile.extendedMaxVelocity = int(args[2])  # 1 or 0, enabled or disabled
                elif (args[0] == 'sensorPosition'):
                    # sensorPosition for x843 family has 3 args
                    if (1):
                        if (len(args) < 4):
                            logger.error("Error: sensorPosition had fewer arguments than expected")
                            continue
                        self.profile.sensorHeight = float(args[1])
                        self.profile.az_tilt = float(args[2])
                        self.profile.elev_tilt = float(args[3])
                        s = SensorPosition()
                        s.parse(args)
                        self._sensor_position = s
                    # sensorPosition for x432 family has 5 args
                    if (0):
                        if (len(args) < 6):
                            logger.error("Error: sensorPosition had fewer arguments than expected")
                            continue
                        # xOffset and yOffset are not implemented in the python code yet.
                        self.profile.xOffset = float(args[1])
                        self.profile.yOffset = float(args[2])
                        self.profile.sensorHeight = float(args[3])
                        self.profile.az_tilt = float(args[4])
                        self.profile.elev_tilt = float(args[5])
                # Only used for Small Obstacle Detection
                elif (args[0] == 'occStateMach'):
                    numZones = int(args[1])
                    if (numZones > 2):
                        logger.error('ERROR: More zones specified by cfg than are supported in this GUI')
                elif (args[0] == 'dfeDataOutputMode'):
                    # this is a chirp config, only mode 1 and 3 are supported
                    self.profile.dfeDataOutputMode = int(args[1])
                    self._sensor._dfeDataOutputMode = int(args[1])
                # Only used for Small Obstacle Detection
                elif (args[0] == 'zoneDef'):
                    if (len(args) < 8):
                        logger.error("Error: zoneDef had fewer arguments than expected")
                        continue
                    zoneIdx = int(args[1])
                    minX = float(args[2])
                    maxX = float(args[3])
                    minY = float(args[4])
                    maxY = float(args[5])
                    # Offset by 3 so it is in center of screen
                    minZ = float(args[6]) + self.profile.sensorHeight
                    maxZ = float(args[7]) + self.profile.sensorHeight

                    name = 'occZone' + str(zoneIdx)

                    self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)
                elif (args[0] == 'mpdBoundaryBox'):
                    if (len(args) < 8):
                        logger.error("Error: mpdBoundaryBox had fewer arguments than expected")
                        continue
                    zoneIdx = int(args[1])
                    minX = float(args[2])
                    maxX = float(args[3])
                    minY = float(args[4])
                    maxY = float(args[5])
                    minZ = float(args[6])
                    maxZ = float(args[7])
                    name = 'mpdBox' + str(zoneIdx)
                    if (OFF_CHIP_PRESENCE_DETECTION_ENABLED == 1):
                        minorMotionStateMachines.append(minorBoundaryBoxStateMachineType())
                    self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)

                elif (args[0] == 'chirpComnCfg'):
                    if (len(args) < 8):
                        logger.error("Error: chirpComnCfg had fewer arguments than expected")
                        continue
                    try:
                        self.chirpComnCfg['DigOutputSampRate'] = int(args[1])
                        self.chirpComnCfg['DigOutputBitsSel'] = int(args[2])
                        self.chirpComnCfg['DfeFirSel'] = int(args[3])
                        self.chirpComnCfg['NumOfAdcSamples'] = int(args[4])
                        self.chirpComnCfg['ChirpTxMimoPatSel'] = int(args[5])
                        self.chirpComnCfg['ChirpRampEndTime'] = 10 * float(args[6])
                        self.chirpComnCfg['ChirpRxHpfSel'] = int(args[7])
                    except Exception as e:
                        print("Exception", e)

                elif (args[0] == 'chirpTimingCfg'):
                    if (len(args) < 6):
                        logger.error("Error: chirpTimingCfg had fewer arguments than expected")
                        continue
                    self.chirpTimingCfg['ChirpIdleTime'] = 10.0 * float(args[1])
                    self.chirpTimingCfg['ChirpAdcSkipSamples'] = int(args[2]) << 10
                    self.chirpTimingCfg['ChirpTxStartTime'] = 10.0 * float(args[3])
                    self.chirpTimingCfg['ChirpRfFreqSlope'] = float(args[4])
                    self.chirpTimingCfg['ChirpRfFreqStart'] = float(args[5])
                elif (args[0] == 'clusterCfg'):
                    if (len(args) < 4):
                        logger.error("Error: clusterCfg had fewer arguments than expected")
                        continue
                    self.profile.enabled = float(args[1])
                    self.profile.maxDistance = float(args[2])
                    self.profile.minPoints = float(args[3])

                # NOTE - Major and Minor motion are not supported at once. Only major or minor motion
                # detection is currently supported.
                elif (args[0] == 'minorStateCfg' or args[0] == 'majorStateCfg'):
                    if (len(args) < 9):
                        logger.error("Error: minorStateCfg had fewer arguments than expected")
                        continue
                    pointThre1 = int(args[1])
                    pointThre2 = int(args[2])
                    snrThre2 = int(args[3])
                    pointHistThre1 = int(args[4])
                    pointHistThre2 = int(args[5])
                    snrHistThre2 = int(args[6])
                    histBufferSize = int(args[7])
                    minor2emptyThre = int(args[8])

                    stateMachineIdx = 0
                    boundaryBoxIdx = 0
                    if (OFF_CHIP_PRESENCE_DETECTION_ENABLED == 1):
                        for box in self.boundaryBoxes:
                            if ('mpdBox' in box['name']):
                                minorMotionStateMachines[stateMachineIdx].configure(
                                    pointThre1, pointThre2, snrThre2, pointHistThre1, pointHistThre2, snrHistThre2, histBufferSize, minor2emptyThre, float(
                                        box['boundList'][0].text()), float(
                                        box['boundList'][1].text()), float(
                                        box['boundList'][2].text()), float(
                                        box['boundList'][3].text()), float(
                                        box['boundList'][4].text()), float(
                                        box['boundList'][5].text()), boundaryBoxIdx)
                                stateMachineIdx = stateMachineIdx + 1
                            boundaryBoxIdx = boundaryBoxIdx + 1
                elif (args[0] == 'cfarFovCfg'):
                    # Command for datapath to filter out detected points outside the
                    # specified limits in the range direction or doppler direction
                    cfar_fov_cfg = CfarFovCfg()
                    cfar_fov_cfg.parse(args)
                    self._cfarFovCfgs.append(cfar_fov_cfg)
                elif (args[0] == 'fovCfg'):
                    fov_cfg = FovCfg()
                    fov_cfg.parse(args)
                    self._fov_cfg = fov_cfg
                elif (args[0] == 'maxAcceleration'):
                    ma = MaxAcceleration()
                    ma.parse(args)
                    self.gtrack.maxAcceleration = ma
                # This is specifically guiMonitor for 60Lo, this parsing will break the gui when an SDK 3 config is sent
                elif (args[0] == 'guiMonitor'):
                    # Plot config message to datapath
                    self._guimonitor = GuiMonitorCfg()
                    self._guimonitor['detectedObjects'] = int(args[2])
                    self._guimonitor['logMagRange'] = int(args[3])
                    self._guimonitor['noiseProfile'] = int(args[4])
                    self._guimonitor['rangeAzimuthElevationHeatMap'] = int(args[5])
                    self._guimonitor['rangeDopplerHeatMap'] = int(args[6])
                    self._guimonitor['statsInfo'] = int(args[7])
                elif (args[0] == 'sensorStart'):
                    pass
                elif (args[0] == 'sensorStop'):
                    pass
                elif (args[0] == 'stateParam'):
                    # tracker
                    state_param = StateParam()
                    state_param.parse(args)
                    self.gtrack.stateParam = state_param
                elif (args[0] == 'flushCfg'):
                    pass
                elif (args[0] == 'channelCfg'):
                    c = ChannelCfg()
                    c.parse(args)
                    self._sensor.set_channel(c)
                elif (args[0] == 'pmicCfg'):
                    pmic_cfg = PmicCfg()
                    pmic_cfg.parse(args)
                    self._pmic_cfg = pmic_cfg
                elif (args[0] == 'vitalsign'):
                    vital_sign_cfg = VitalSignCfg()
                    vital_sign_cfg.parse(args)
                    self._vital_sign_cfg = vital_sign_cfg
                elif (args[0] == 'VSRangeIdxCfg'):
                    vs_range_idx_cfg = VSRangeIdxCfg()
                    vs_range_idx_cfg.parse(args)
                    self._vs_range_idx_cfg = vs_range_idx_cfg
                elif (args[0] == 'advFrameCfg'):
                    adv_frame_cfg = AdvFrameCfg()
                    adv_frame_cfg.parse(args)
                    self._adv_frame_cfg = adv_frame_cfg
                elif (args[0] == 'subFrameCfg'):
                    sub_frame_cfg = SubFrameCfg()
                    sub_frame_cfg.parse(args)
                    if not hasattr(self, '_sub_frame_cfgs'):
                        self._sub_frame_cfgs = []
                    self._sub_frame_cfgs.append(sub_frame_cfg)
                elif (args[0] == 'aoaFovCfg'):
                    aoa_fov_cfg = AoaFovCfg()
                    aoa_fov_cfg.parse(args)
                    self._aoa_fov_cfg = aoa_fov_cfg
                elif (args[0] == 'measureRangeBiasAndRxChanPhase'):
                    measure_cfg = MeasureRangeBiasAndRxChanPhase()
                    measure_cfg.parse(args)
                    self._measure_range_bias_cfg = measure_cfg
                elif (args[0] == 'calibData'):
                    calib_data = CalibData()
                    calib_data.parse(args)
                    self._calib_data = calib_data
                elif (args[0] == 'CQRxSatMonitor'):
                    cq_rx_sat_monitor = CQRxSatMonitor()
                    cq_rx_sat_monitor.parse(args)
                    if not hasattr(self, '_cq_rx_sat_monitors'):
                        self._cq_rx_sat_monitors = []
                    self._cq_rx_sat_monitors.append(cq_rx_sat_monitor)
                elif (args[0] == 'CQSigImgMonitor'):
                    cq_sig_img_monitor = CQSigImgMonitor()
                    cq_sig_img_monitor.parse(args)
                    if not hasattr(self, '_cq_sig_img_monitors'):
                        self._cq_sig_img_monitors = []
                    self._cq_sig_img_monitors.append(cq_sig_img_monitor)
                else:
                    if args[0][0] != '%':
                        if self._show_warning:
                            logger.warning("UNKNOWN ATH {}", args)

                        pass

            counter += 1

        # self.rangeRes = (3e8*(100/self.chirpComnCfg['DigOutputSampRate']))/(2*self.chirpTimingCfg['ChirpRfFreqSlope']*self.chirpComnCfg['NumOfAdcSamples'])
        self.rangeRes = (3e8 * (100 / self.chirpComnCfg['DigOutputSampRate']) * 1e6) / (
            2 * self.chirpTimingCfg['ChirpRfFreqSlope'] * 1e12 * self.chirpComnCfg['NumOfAdcSamples'])
        # print("self.rangePlot.setXRange(0,(self.chirpComnCfg['NumOfAdcSamples']/2)*self.rangeRes,padding=0.01)")

        self.rangeAxisVals = np.arange(0, self.chirpComnCfg['NumOfAdcSamples'] / 2 * self.rangeRes, self.rangeRes)
        self.profile.rangeIdxToMeters = (299792458 * self.profile.sampleRate * 1e3) / \
            (2 * self.profile.slope * 1e12 * self._profiles[0].num_range_bins)
        self.profile.maxVelocity = 299792458 / (4 * self.profile.startFreq * 1e9 * (
            self.profile.idle + self.profile.rampEnd) * 1e-6 * self._sensor.num_tx_antennas)

    def sensor(self) -> RadarSS:
        return self._sensor

    def to_dict(self) -> dict:
        config_dict = {
            "num_rx_antennas": self._sensor.num_rx_antennas,
            "num_tx_antennas": self._sensor.num_tx_antennas,


            "num_diff_chirps": len(self._sensor._chirps),
            "idle_time": self._profiles[0].idleTime,
            "start_frequency": self._profiles[0].startFreq,
            "frequency_slope": self._profiles[0].freqSlopeConst,
            "ramp_end_time": self._profiles[0].rampEndTime,

            "num_adc_samples": self._sensor.num_adc_samples,
            "adc_sampling_frequency": self._profiles[0].digOutSampleRate,

            "num_loops": self._sensor._frames[0].num_loops,
            "num_chirps_per_frame": self._sensor.num_chirps_per_frame,

            "num_range_bins": self._sensor.num_range_bins,
            "range_max": np.round(self._sensor.range_max, 2),
            "range_resolution": np.round(self._sensor.range_resolution, 2),

            "num_doppler_bins": self._sensor.num_doppler_bins,
            "max_velocity": np.round(self._sensor.max_velocity, 2),
            "velocity_resolution": np.round(self._sensor.doppler_resolution, 2),

            "frame_periodicity": self._frames[0].framePeriodicity,
            "fps": int(1000 / self._frames[0].framePeriodicity),
            "clutter_removal_enabled": [None if self._clutter_removal is None else self._clutter_removal.enabled()][0],
            "cfar_db_threshold": self._cfarCfgs[0].thresholdScale if self._cfarCfgs else None
        }
        return config_dict
