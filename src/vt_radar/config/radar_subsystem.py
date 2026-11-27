"""
Radar Subsystem Configuration Classes

This module contains the core radar hardware configuration classes for the TI mmWave SDK.
These classes represent the fundamental radar subsystem components including ADC, antennas,
chirps, profiles, frames, and signal processing configurations.

Based on TI mmWave SDK documentation.
"""

from loguru import logger
from typing import Any, List
import numpy as np
import tabulate
from .base import Cfg


class ChannelCfg(Cfg):
    """
    Channel configuration for RX/TX antennas.

    This configuration defines which RX and TX antenna channels are enabled
    for the radar system. It controls the physical antenna configuration
    and cascading mode for multi-chip setups.

    Parameters:
        rxChannelEn (int): RX channel enable mask (bit 0-3 for RX0-RX3)
        txChannelEn (int): TX channel enable mask (bit 0-2 for TX0-TX2)
        cascading (int): Cascading mode (0=disabled, 1=enabled)

    Valid Values:
        - rxChannelEn: 0-15 (4-bit mask)
        - txChannelEn: 0-7 (3-bit mask)
        - cascading: 0-1

    Example:
        channelCfg 15 7 0  # Enable all RX channels, all TX channels, no cascading
    """

    def __init__(self) -> None:
        self.rxChannelEn: int = 0
        self.txChannelEn: int = 0
        self.cascading: int = 0

    def describe(self) -> None:
        table = [
            ["RX channel enabled", antenna_enabled(self.rxChannelEn)],
            ["TX channel enabled", antenna_enabled(self.txChannelEn)],
            ["Cascading", self.cascading]
        ]
        print(tabulate.tabulate(table))

    @property
    def num_rx_antennas(self) -> int:
        return len(antenna_enabled(self.rxChannelEn))

    @property
    def num_tx_antennas(self) -> int:
        return len(antenna_enabled(self.txChannelEn))

    def parse(self, args: List[str]) -> None:
        self.rxChannelEn = int(args[1])
        self.txChannelEn = int(args[2])
        self.cascading = int(args[3])  # N/A


class AdcCfg(Cfg):
    """
    ADC (Analog-to-Digital Converter) configuration parameters.

    The ADC configuration defines the number of ADC bits and output format
    for the radar data processing chain. This is a critical configuration
    that affects the dynamic range and data throughput of the radar system.

    Parameters:
        numADCBits (int): Number of ADC bits
            - 0: 12-bit ADC
            - 1: 14-bit ADC
            - 2: 16-bit ADC (recommended for mmWave demo)
        adcOutputFmt (int): ADC output format
            - 0: Real output (not supported in mmWave demo)
            - 1: Complex 1x output (recommended)
            - 2: Complex 2x output (higher throughput)

    Valid Values:
        - numADCBits: 0-2
        - adcOutputFmt: 0-2

    Constraints:
        - mmWave demo only supports 16-bit ADC (numADCBits=2)
        - mmWave demo only supports complex output formats (adcOutputFmt=1 or 2)

    Example:
        adcCfg 2 1  # 16-bit ADC with complex 1x output
    """

    def __init__(self) -> None:
        self.numADCBits: int = 2
        self.adcOutputFmt: int = 1

    def parse(self, args: List[str]) -> None:
        """
        Parse ADC configuration parameters from command line arguments.

        Args:
            args (list): Command line arguments containing ADC parameters
        """
        if len(args) < 3:
            raise ValueError("adcCfg requires at least 3 arguments")

        self.numADCBits = int(args[1])
        self.adcOutputFmt = int(args[2])

        # Range checks based on TI documentation
        if self.numADCBits not in [0, 1, 2]:
            raise ValueError("numADCBits must be 0 (12-bits), 1 (14-bits), or 2 (16-bits)")
        if self.adcOutputFmt not in [0, 1, 2]:
            raise ValueError("adcOutputFmt must be 0 (real), 1 (complex 1x), or 2 (complex 2x)")

        # Additional validation based on TI documentation
        if self.numADCBits != 2:
            raise ValueError("mmWave demo supports only 16-bit ADC (numADCBits=2)")
        if self.adcOutputFmt == 0:
            raise ValueError("mmWave demo supports only complex output formats (adcOutputFmt=1 or 2)")

    def describe(self) -> None:
        """Print a formatted description of the ADC configuration."""
        adc_bits_map = {0: "12 bits", 1: "14 bits", 2: "16 bits"}
        output_fmt_map = {0: "Real", 1: "Complex 1x", 2: "Complex 2x"}

        table = [
            ["ADC Bits", adc_bits_map[self.numADCBits]],
            ["Output Format", output_fmt_map[self.adcOutputFmt]],
            ["Supported in Demo", "Yes" if self.numADCBits == 2 and self.adcOutputFmt in [1, 2] else "No"]
        ]
        print(tabulate.tabulate(table))


class AdcBufCfg(Cfg):
    """
    ADC Buffer configuration for data buffering and formatting.

    This configuration controls how ADC data is buffered and formatted
    before being sent to the processing chain. It includes settings for
    subframe indexing, output formatting, and data interleaving.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        adcOutputFmt (int): ADC output format (0=real, 1=complex 1x, 2=complex 2x)
        sampleSwap (int): Sample swap enable (only 1 supported)
        chanInterleave (int): Channel interleave enable (only 1 supported)
        chirpThreshold (int): Chirp threshold for data processing

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - adcOutputFmt: 0-2
        - sampleSwap: 1 (only supported value)
        - chanInterleave: 1 (only supported value)
        - chirpThreshold: 0-255

    Constraints:
        - sampleSwap must be 1 (only option supported)
        - chanInterleave must be 1 (only option supported)

    Example:
        adcbufCfg -1 0 1 1 1  # All subframes, real output, swap enabled, interleave enabled
    """

    def describe(self) -> None:
        pass

    def parse(self, args: List[str]) -> None:
        if len(args) < 6:
            raise ValueError("adcbufCfg requires at least 6 arguments")
        self.subFrameIdx = int(args[1])
        self.adcOutputFmt = int(args[2])
        self.sampleSwap = int(args[3])  # only option 1 is supported
        self.chanInterleave = int(args[4])  # only option 1 is supported
        self.chirpThreshold = int(args[5])

        # Validate supported options
        if self.sampleSwap != 1:
            raise ValueError("sampleSwap must be 1 (only option supported)")
        if self.chanInterleave != 1:
            raise ValueError("chanInterleave must be 1 (only option supported)")


class AntGeometry(Cfg):
    """
    Antenna geometry configuration for virtual antenna positioning.

    This configuration defines the physical location indices of virtual antennas
    in the azimuth and elevation domains. It is used for beamforming and
    angle of arrival estimation algorithms.

    Parameters:
        virtAntIdx0-11 (int): Virtual antenna indices for 12 virtual antennas
            - Values represent physical antenna spacing in half-wavelength units
            - Negative values indicate antennas behind the reference
            - Zero indicates the reference antenna position

    Valid Values:
        - virtAntIdx0-11: Integer values representing antenna positions
        - Typical range: -3 to +3 (depending on antenna array design)

    Usage:
        - antGeometry0: Azimuth domain antenna positions
        - antGeometry1: Elevation domain antenna positions

    Example:
        antGeometry0 -1 -1 0 0 -3 -3 -2 -2 -1 -1 0 0
        antGeometry1 -1 0 -1 0 -3 -2 -3 -2 -3 -2 -3 -2
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 13:
            raise ValueError("antGeometry requires at least 13 arguments")
        self.virtAntIdx0 = int(args[1])
        self.virtAntIdx1 = int(args[2])
        self.virtAntIdx2 = int(args[3])
        self.virtAntIdx3 = int(args[4])
        self.virtAntIdx4 = int(args[5])
        self.virtAntIdx5 = int(args[6])
        self.virtAntIdx6 = int(args[7])
        self.virtAntIdx7 = int(args[8])
        self.virtAntIdx8 = int(args[9])
        self.virtAntIdx9 = int(args[10])
        self.virtAntIdx10 = int(args[11])
        self.virtAntIdx11 = int(args[12])


class AntPhaseRot(Cfg):
    """
    Antenna phase rotation configuration for board design compensation.

    This configuration compensates for phase rotations introduced by the
    board design. Each antenna element can have a phase rotation that
    needs to be corrected for accurate beamforming and angle estimation.

    Parameters:
        virtAntIdx0-11 (int): Phase rotation compensation for 12 virtual antennas
            - 1: No phase rotation (normal case)
            - -1: 180-degree phase rotation (board design compensation)

    Valid Values:
        - virtAntIdx0-11: 1 or -1 only

    Usage:
        - Used to compensate for board-level phase inversions
        - Critical for accurate angle of arrival estimation
        - Must match the actual hardware board design

    Example:
        antPhaseRot 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 13:
            raise ValueError("antPhaseRot requires at least 13 arguments")
        self.virtAntIdx0 = int(args[1])
        self.virtAntIdx1 = int(args[2])
        self.virtAntIdx2 = int(args[3])
        self.virtAntIdx3 = int(args[4])
        self.virtAntIdx4 = int(args[5])
        self.virtAntIdx5 = int(args[6])
        self.virtAntIdx6 = int(args[7])
        self.virtAntIdx7 = int(args[8])
        self.virtAntIdx8 = int(args[9])
        self.virtAntIdx9 = int(args[10])
        self.virtAntIdx10 = int(args[11])
        self.virtAntIdx11 = int(args[12])


class ChirpCfg(Cfg):
    """
    Chirp configuration for radar signal generation.

    This configuration defines the parameters for individual chirps in the
    radar frame. Each chirp represents a frequency sweep from start frequency
    to end frequency over a specific time period.

    Parameters:
        startIdx (int): Start chirp index (must equal endIdx for single chirp)
        endIdx (int): End chirp index (must equal startIdx for single chirp)
        profileId (int): Profile ID to use for this chirp (0-3)
        startFreq (float): Start frequency in GHz
        freqSlope (float): Frequency slope in MHz/us
        idleTime (float): Idle time between chirps in us
        adcStartTime (float): ADC start time in us
        txAntMask (int): TX antenna mask (bit 0-2 for TX0-TX2)

    Valid Values:
        - startIdx, endIdx: 0-2 (must be equal)
        - profileId: 0-3
        - startFreq: 60-81 GHz (typical range)
        - freqSlope: 0-1000 MHz/us (typical range)
        - idleTime: 0-1000 us
        - adcStartTime: 0-1000 us
        - txAntMask: 0-7 (3-bit mask)

    Constraints:
        - startIdx must equal endIdx
        - Individual chirps should have only one distinct TX antenna enabled (MIMO)
        - Same TX antennas should be enabled for all chirps in non-MIMO mode

    Example:
        chirpCfg 0 0 0 60.0 50.0 7.0 7.0 1  # Single chirp with TX0
    """

    def __init__(self) -> None:
        self.startIdx: int = 0
        self.endIdx: int = 0
        self.profileId: int = 0
        self.startFreq: float = 60.0
        self.freqSlope: float = 50.0
        self.idleTime: float = 7.0
        self.adcStartTime: float = 7.0
        self.txAntMask: int = 1

    def __eq__(self, other: 'ChirpCfg') -> bool:
        return self.freqSlope == other.freqSlope and \
            self.idleTime == other.idleTime

    def __str__(self):
        return "C" + str(self.freqSlope)

    def describe(self) -> None:
        table = [
            ["start_idx", self.startIdx],
            ["end_idx", self.endIdx],
            ["profile_id", self.profileId],
            ["start_freq", self.startFreq],
            ["freq_slope", self.freqSlope],
            ["idle_time", self.idleTime],
            ["adc_start_time", self.adcStartTime],
            ["tx_ant_mask", antenna_enabled(self.txAntMask)]
        ]
        print(tabulate.tabulate(table))

    def doc(self):
        print("idle_time: The time between the end of previous chirp and start of next chirp")
        print("profile_id: ")
        print("tx_ant_mask: Individual chirps should have either only one distinct Tx antenna enabled (MIMO) or same TX antennas should be enabled for all chirps")

    def parse(self, args: List[str]) -> None:
        self.startIdx = int(args[1])
        self.endIdx = int(args[2])
        self.profileId = int(args[3])
        self.startFreq = float(args[4])
        self.freqSlope = float(args[5])
        self.idleTime = float(args[6])
        self.adcStartTime = float(args[7])
        self.txAntMask = int(args[8])
        assert (self.startIdx == self.endIdx)
        assert (self.startIdx in [0, 1, 2])
        assert (self.endIdx in [0, 1, 2])


class ProfileCfg(Cfg):
    """
    Profile configuration for radar chirp parameters.

    This configuration defines the fundamental parameters for radar chirps,
    including frequency settings, timing, and signal processing parameters.
    Multiple profiles can be defined and referenced by chirp configurations.

    Parameters:
        profileId (int): Profile ID (0-3)
        startFreq (float): Start frequency in GHz
        idleTime (float): Idle time in us
        adcStartTime (float): ADC start time in us
        rampEndTime (float): Ramp end time in us
        txOutPower (float): TX output power (typically 0)
        txPhaseShifter (float): TX phase shifter (typically 0)
        freqSlopeConst (float): Frequency slope constant in MHz/us
        txStartTime (float): TX start time in us
        numAdcSamples (float): Number of ADC samples
        digOutSampleRate (float): Digital output sample rate in ksps
        hpfCornerFreq1 (int): High-pass filter corner frequency 1 (0-3)
        hpfCornerFreq2 (int): High-pass filter corner frequency 2 (0-3)
        rxGain (str): RX gain setting

    Valid Values:
        - profileId: 0-3
        - startFreq: 60-81 GHz
        - idleTime: 0-1000 us
        - adcStartTime: 0-1000 us
        - rampEndTime: 0-1000 us
        - txOutPower: 0 (fixed)
        - txPhaseShifter: 0 (fixed)
        - freqSlopeConst: 0-1000 MHz/us
        - txStartTime: 0-1000 us
        - numAdcSamples: 64-4096
        - digOutSampleRate: 1000-20000 ksps
        - hpfCornerFreq1,2: 0-3
        - rxGain: String value

    Example:
        profileCfg 0 60.75 30.00 25.00 59.10 0 0 54.71 1 96 2950.00 2 1 36
    """

    def __eq__(self, other):
        return self.startFreq == other.startFreq and \
            self.freqSlopeConst == other.freqSlopeConst

    def __getitem__(self, __name: str) -> Any:
        # Attributes are stored without an underscore prefix (parse() sets self.startFreq,
        # etc.), so "_" + name looked up a nonexistent key and KeyError'd every access.
        return self.__dict__[__name]

    def __sub__(self, other):
        a = self.table()
        b = other.table()

        result = []
        for x1, x2 in zip(a, b):
            if x1[1] != x2[1]:
                result.append([x1[0], x1[1], x2[1]])
        if len(result):
            return tabulate.tabulate(result)

    def describe(self) -> None:
        print(tabulate.tabulate(self.table()))
        print("THUS: ADC Sampling Time", self.numAdcSamples / self.digOutSampleRate)
        print("THUS: Sweep bandwidth", self.freqSlopeConst * (self.numAdcSamples / self.digOutSampleRate))
        print()

    def parse(self, args: List[str]) -> None:
        # Reads args[1..14] below; guard like the sibling cfg classes
        # (BoundaryBox, FovCfg, calibData) instead of IndexError'ing on a
        # truncated line.
        if len(args) < 15:
            raise ValueError("profileCfg requires at least 15 arguments")
        self.profileId = int(args[1])  # 0-3
        self.startFreq = float(args[2])  # eg 77, 61.38
        self.idleTime = float(args[3])  # eg 7, 7.15
        self.adcStartTime = float(args[4])  # eg 7, 7.34
        self.rampEndTime = float(args[5])  # eg 58, 216.15
        self.txOutPower = float(args[6])  # always 0
        self.txPhaseShifter = float(args[7])  # always 0
        self.freqSlopeConst = float(args[8])  # 68, 16.83
        self.txStartTime = float(args[9])  # 1, 2.92
        self.numAdcSamples = float(args[10])  # 256, 224
        self.digOutSampleRate = float(args[11]) / 1e3  # 5500 (directly convert from k to M)
        self.hpfCornerFreq1 = int(args[12])  # 0-3
        self.hpfCornerFreq2 = int(args[13])  # 0-3
        self.rxGain = args[14]

    @property
    def dig_out_sample_rate(self) -> float:
        """
            returns the sample rate in Ms/s
        """
        return self.digOutSampleRate

    @property
    def num_adc_samples(self) -> int:
        return int(self.numAdcSamples)

    @property
    def num_range_bins(self) -> int:
        # in the chip it is rounded to 2, however the datacube size
        # depends on the number of adc samples actually returned
        # return int(self._num_adc_samples)

        # the nr range bins is ceil(log2(nrAdcSamples))
        numAdcSamplesRoundTo2 = 1
        while self.numAdcSamples > numAdcSamplesRoundTo2:
            numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
        return numAdcSamplesRoundTo2

    def table(self):
        table = [
            ["Profile", "#" + str(self.profileId)],
            ["Start frequency ", str(self.startFreq) + " GHz"],
            ["Frequency Slope ", str(self.freqSlopeConst) + " MHz/us"],
            ["ADC sampling frequency", str(self.digOutSampleRate) + " Ms/s"],
            ["Samples per chirp ", self.numAdcSamples],
            ["Idle Time ", str(self.idleTime) + " us"],
            ["ADC Valid Start Time", self.adcStartTime],
            ["Ramp End Time", str(self.rampEndTime) + " us"],
            ["TX Start Time", self.txStartTime],
            # ["TX power", self.txOutPower], # tx power is not used
            # ["TX phase", self.txPhaseShifter]
        ]
        return table


class FrameCfg(Cfg):
    """
    Frame configuration for radar timing and structure.

    This configuration defines the overall frame structure including
    chirp indices, number of loops, frames, and timing parameters.
    It controls the basic radar operation timing.

    Parameters:
        chirpStartIdx (int): Start chirp index (0-2)
        chirpEndIdx (int): End chirp index (0-2)
        numLoops (int): Number of loops per frame (multiple of 4)
        numFrames (int): Number of frames (0 for continuous)
        framePeriodicity (float): Frame periodicity in ms
        triggerSelect (int): Trigger select (0=internal, 1=external)
        frameTriggerDelay (float): Frame trigger delay in us

    Valid Values:
        - chirpStartIdx, chirpEndIdx: 0-2
        - numLoops: 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64
        - numFrames: 0 (continuous) or positive integer
        - framePeriodicity: 10-1000 ms
        - triggerSelect: 0-1
        - frameTriggerDelay: 0-1000 us

    Constraints:
        - numLoops should be a multiple of 4
        - chirpEndIdx should be >= chirpStartIdx

    Example:
        frameCfg 0 2 48 0 55.00 1 0  # 48 loops, continuous, 55ms period
    """

    def describe(self) -> None:
        table = [
            #  ["ChirpStartIndex", self.chirpStartIdx],
            #    ["ChirpEndIndex", self.chirpEndIdx],
            ["NumberOfLoops", self.numLoops]
        ]
        print(tabulate.tabulate(table))

    def doc(self):
        # number of loops should be a multiple of 4
        print("NumberOfLoops: the number of chirps per frame")

    @property
    def num_loops(self):
        return self.numLoops

    def parse(self, args: List[str]) -> None:
        # Needs args[1..7] below, i.e. 8 tokens including the command name.
        # The old `< 4` guard let 4-7 token lines through and then IndexError'd
        # on args[4].
        if (len(args) < 8):
            print("Error: frameCfg had fewer arguments than expected")
            return False
        self.chirpStartIdx = int(args[1])
        self.chirpEndIdx = int(args[2])
        self.numLoops = int(args[3])
        self.numFrames = int(args[4])
        self.framePeriodicity = float(args[5])
        self.triggerSelect = int(args[6])
        self.frameTriggerDelay = float(args[7])

        return True


class ClutterRemoval(Cfg):
    """
    Static clutter removal algorithm configuration.

    This configuration enables static clutter removal by subtracting
    the mean value of input samples from the 2D-FFT. This helps
    remove stationary objects and improve detection of moving targets.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        enabled (int): Enable clutter removal (0=disabled, 1=enabled)

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - enabled: 0-1

    Algorithm:
        - Calculates mean of input samples
        - Subtracts mean from 2D-FFT input
        - Helps remove stationary clutter
        - Improves moving target detection

    Example:
        clutterRemoval -1 1  # Enable for all subframes
    """

    def parse(self, args: List[str]) -> None:
        self.subFrameIdx = int(args[1])
        # Store in a separate field so it does not shadow the enabled() method.
        # (self.enabled = int(...) previously overwrote the method, so a later
        # self._clutter_removal.enabled() raised "'int' object is not callable".)
        self.enabled_flag = int(args[2])

    def enabled(self) -> bool:
        return bool(self.enabled_flag)


class CalibDcRangeSig(Cfg):
    """
    DC range signature calibration configuration.

    This configuration enables DC range signature calibration to remove
    antenna coupling effects that dominate range bins close to the radar.
    The signature is estimated during the first N chirps and then
    subtracted from subsequent chirps.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        enabled (int): Enable DC calibration (0=disabled, 1=enabled)
        negativeBinIdx (int): Negative bin index for DC signature estimation
        positiveBinIdx (int): Positive bin index for DC signature estimation
        numAvg (int): Number of chirps for averaging (typically 256)

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - enabled: 0-1
        - negativeBinIdx: -10 to 0
        - positiveBinIdx: 0 to 10
        - numAvg: 1-1000 (typically 256)

    Algorithm:
        - Estimates DC signature during first N chirps
        - Accumulates and averages specified bins around DC
        - Subtracts estimated signature from subsequent chirps
        - Assumes no objects present during estimation period

    Example:
        calibDcRangeSig -1 1 -3 3 256  # Enable, bins -3 to +3, 256 chirps
    """

    def parse(self, args: List[str]) -> None:
        self.subFrameIdx = int(args[1])
        self.enabled = int(args[2])
        self.negativeBinIdx = int(args[3])
        self.positiveBinIdx = int(args[4])
        # Value of 256 means first 256 chirps (after command is issued and feature is enabled) will be used for
        # collecting (averaging) DC signature in the bins specified above. From 257th chirp, the collected DC
        # signature will be removed from every chirp.
        self.numAvg = int(args[5])


class CompRangeBiasAndRxChanPhase(Cfg):
    """
    Range bias and RX channel phase compensation configuration.

    This configuration provides compensation values for range bias and
    RX channel phase corrections. These values are used to correct
    for hardware imperfections and improve measurement accuracy.

    Parameters:
        re00-re10 (float): Real part compensation for RX channels 0-10
        im00-im10 (float): Imaginary part compensation for RX channels 0-10

    Valid Values:
        - re00-re10: Float values (typically -1.0 to 1.0)
        - im00-im10: Float values (typically -1.0 to 1.0)

    Usage:
        - Compensates for range bias in distance measurements
        - Corrects RX channel phase differences
        - Improves angle of arrival estimation accuracy
        - Calibrated values should be obtained from measurement

    Example:
        compRangeBiasAndRxChanPhase 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
    """

    def parse(self, args: List[str]) -> None:
        # All re/im entries are float compensation values (typically -1.0..1.0).
        # Only re00 was float; the rest used int(), which truncates fractional
        # calibration values and raises ValueError on strings like "-0.5".
        self.re00 = float(args[1])
        self.im00 = float(args[2])
        self.re01 = float(args[3])
        self.im01 = float(args[4])
        self.re02 = float(args[5])
        self.im02 = float(args[6])
        self.re03 = float(args[7])
        self.im03 = float(args[8])
        self.re04 = float(args[9])
        self.im04 = float(args[10])
        self.re05 = float(args[11])
        self.im05 = float(args[12])
        self.re06 = float(args[13])
        self.im06 = float(args[14])
        self.re07 = float(args[15])
        self.im07 = float(args[16])
        self.re08 = float(args[17])
        self.im08 = float(args[18])
        self.re09 = float(args[19])
        self.im09 = float(args[20])
        self.re10 = float(args[21])
        self.im10 = float(args[22])
        #   self.re11 = int(args[23])
        #   self.im11 = int(args[24])
        #   self.re12 = int(args[25])
        #   self.im12 = int(args[26])


class LvdsStreamCfg(Cfg):
    """
    LVDS stream configuration for data output.

    This configuration controls the LVDS (Low Voltage Differential Signaling)
    data stream output from the radar chip. It defines how radar data
    is formatted and transmitted over the LVDS interface.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        enableHeader (int): Enable header in data stream (0=disabled, 1=enabled)
        dataFmt (int): Data format (0=real, 1=complex 1x, 2=complex 2x)
        enableSW (int): Enable software control (0=disabled, 1=enabled)

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - enableHeader: 0-1
        - dataFmt: 0-2
        - enableSW: 0-1

    Usage:
        - Controls data output format over LVDS interface
        - Enables/disables header information in data stream
        - Configures data format for downstream processing
        - Software control for dynamic configuration changes

    Example:
        LvdsStreamCfg -1 0 0 0  # All subframes, no header, real data, no SW control
    """

    def parse(self, args: List[str]) -> None:
        self.subFrameIdx = int(args[1])
        self.enableHeader = int(args[2])
        self.dataFmt = int(args[3])
        self.enableSW = int(args[4])


class AdvFrameCfg(Cfg):
    """
    Advanced frame configuration for multi-subframe radar operation.

    This configuration enables complex frame structures with multiple
    subframes, allowing for different radar parameters per subframe.
    This is useful for applications requiring different range/velocity
    resolutions or different detection algorithms per subframe.

    Parameters:
        numSubFrames (int): Number of subframes (1-8)
        numLoops (int): Number of loops per subframe
        numFrames (int): Number of frames (0 for continuous)
        framePeriodicity (float): Frame periodicity in ms
        triggerSelect (int): Trigger select (0=internal, 1=external)

    Valid Values:
        - numSubFrames: 1-8
        - numLoops: 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64
        - numFrames: 0 (continuous) or positive integer
        - framePeriodicity: 10-1000 ms
        - triggerSelect: 0-1

    Usage:
        - Enables multi-subframe radar operation
        - Each subframe can have different parameters
        - Useful for multi-resolution applications
        - Requires subFrameCfg for each subframe

    Example:
        advFrameCfg 4 0 0 1 0  # 4 subframes, continuous, 1ms period, internal trigger
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 6:
            raise ValueError("advFrameCfg requires at least 6 arguments")
        self.numSubFrames = int(args[1])
        self.numLoops = int(args[2])
        self.numFrames = int(args[3])
        self.framePeriodicity = float(args[4])
        self.triggerSelect = int(args[5])


class SubFrameCfg(Cfg):
    """
    Sub-frame configuration for advanced radar frame structures.

    This configuration defines parameters for individual subframes within
    a multi-subframe radar configuration. Each subframe can have different
    chirp parameters, timing, and processing settings.

    Parameters:
        subFrameIdx (int): Subframe index (0-7)
        startChirpIdx (int): Start chirp index for this subframe
        endChirpIdx (int): End chirp index for this subframe
        numLoops (int): Number of loops for this subframe
        numFrames (int): Number of frames for this subframe
        framePeriodicity (float): Frame periodicity in ms
        triggerSelect (int): Trigger select (0=internal, 1=external)
        frameTriggerDelay (float): Frame trigger delay in us
        numSubFrames (int): Total number of subframes

    Valid Values:
        - subFrameIdx: 0-7
        - startChirpIdx, endChirpIdx: 0-2
        - numLoops: 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64
        - numFrames: 0 (continuous) or positive integer
        - framePeriodicity: 10-1000 ms
        - triggerSelect: 0-1
        - frameTriggerDelay: 0-1000 us
        - numSubFrames: 1-8

    Usage:
        - Defines individual subframe parameters
        - Used with advFrameCfg for multi-subframe operation
        - Each subframe can have different chirp sequences
        - Enables flexible radar operation modes

    Example:
        subFrameCfg 0 0 0 2 32 100 0 1 1 100  # Subframe 0, chirps 0-0, 32 loops
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 10:
            raise ValueError("subFrameCfg requires at least 10 arguments")
        self.subFrameIdx = int(args[1])
        self.startChirpIdx = int(args[2])
        self.endChirpIdx = int(args[3])
        self.numLoops = int(args[4])
        self.numFrames = int(args[5])
        self.framePeriodicity = float(args[6])
        self.triggerSelect = int(args[7])
        self.frameTriggerDelay = float(args[8])
        self.numSubFrames = int(args[9])


class AoaFovCfg(Cfg):
    """
    Angle of Arrival (AoA) Field of View configuration.

    This configuration defines the field of view parameters for angle of
    arrival estimation in both azimuth and elevation domains. It sets
    the angular limits for beamforming and angle estimation algorithms.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        azimuthMin (float): Minimum azimuth angle in degrees
        azimuthMax (float): Maximum azimuth angle in degrees
        elevationMin (float): Minimum elevation angle in degrees
        elevationMax (float): Maximum elevation angle in degrees

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - azimuthMin: -90 to 90 degrees
        - azimuthMax: -90 to 90 degrees
        - elevationMin: -90 to 90 degrees
        - elevationMax: -90 to 90 degrees

    Constraints:
        - azimuthMin < azimuthMax
        - elevationMin < elevationMax
        - Typical range: -90 to +90 degrees for both domains

    Usage:
        - Defines angular search space for AoA estimation
        - Limits beamforming to specified angular range
        - Improves processing efficiency by reducing search space
        - Critical for accurate angle estimation

    Example:
        aoaFovCfg -1 -90 90 -90 90  # Full 360-degree coverage
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 6:
            raise ValueError("aoaFovCfg requires at least 6 arguments")
        self.subFrameIdx = int(args[1])
        self.azimuthMin = float(args[2])
        self.azimuthMax = float(args[3])
        self.elevationMin = float(args[4])
        self.elevationMax = float(args[5])


class MeasureRangeBiasAndRxChanPhase(Cfg):
    """
    Range bias and RX channel phase measurement configuration.

    This configuration is used for calibration measurements to determine
    range bias and RX channel phase corrections. It enables the radar
    to measure and compensate for hardware imperfections.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        rangeBias (float): Range bias measurement in meters
        phaseCorrection (float): Phase correction measurement in radians

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - rangeBias: -10.0 to 10.0 meters
        - phaseCorrection: -3.14 to 3.14 radians

    Usage:
        - Enables range bias measurement and correction
        - Measures RX channel phase differences
        - Improves measurement accuracy
        - Used during calibration procedures

    Example:
        measureRangeBiasAndRxChanPhase -1 0.1 0.2  # 0.1m bias, 0.2rad phase
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 3:
            raise ValueError("measureRangeBiasAndRxChanPhase requires at least 3 arguments")
        self.subFrameIdx = int(args[1])
        self.rangeBias = float(args[2])
        self.phaseCorrection = float(args[3]) if len(args) > 3 else 0.0


class CalibData(Cfg):
    """
    Calibration data configuration.

    This configuration stores calibration data for radar system compensation.
    It includes various calibration parameters needed for accurate radar
    measurements and signal processing.

    Parameters:
        subFrameIdx (int): Subframe index (-1 for all subframes)
        dataType (int): Type of calibration data
        dataLength (int): Length of calibration data

    Valid Values:
        - subFrameIdx: -1 or 0-7
        - dataType: 0-255 (calibration data type identifier)
        - dataLength: 1-10000 (number of calibration data points)

    Usage:
        - Stores system calibration parameters
        - Used for radar system compensation
        - Improves measurement accuracy
        - Critical for production calibration

    Example:
        calibData 0 0 0  # Subframe 0, data type 0, length 0
    """

    def parse(self, args: List[str]) -> None:
        if len(args) < 4:
            raise ValueError("calibData requires at least 4 arguments")
        self.subFrameIdx = int(args[1])
        self.dataType = int(args[2])
        self.dataLength = int(args[3])
        # Additional calibration data parameters can be added as needed


def antenna_enabled(mask) -> List[int]:
    """Helper function to get enabled antenna channels from mask"""
    en = []
    for channel in range(4):  # MAX RX
        if mask & (0x01 << channel):
            en.append(channel + 1)
    return en


class RadarSS(object):
    """
    The radar subsystem - core hardware signal processing chain

    This class represents the fundamental radar subsystem components including
    ADC, antennas, chirps, profiles, frames, and signal processing configurations.
    Based on TI mmWave SDK documentation.
    """

    def __init__(self):
        self._channel: ChannelCfg = None
        self._chirps: List[ChirpCfg] = []
        self._profiles: List[ProfileCfg] = []
        self._frames: List[FrameCfg] = []
        self._adc_cfg: AdcCfg = None
        self._adc_buf_cfg: AdcBufCfg = None
        self._ant_geometry0: AntGeometry = None
        self._ant_geometry1: AntGeometry = None
        self._ant_phase_rot: AntPhaseRot = None
        self._clutter_removal: ClutterRemoval = None
        self._calib_dc_range_sig: CalibDcRangeSig = None
        self._comp: CompRangeBiasAndRxChanPhase = None
        self._lvds_stream_cfg: LvdsStreamCfg = None

    def __eq__(self, other):
        ret = True

        for p1, p2 in zip(self._profiles, other._profiles):
            ret &= (p1 == p2)
        for c1, c2 in zip(self._chirps, other._chirps):
            ret &= (c1 == c2)

        return ret

    def __sub__(self, other):
        p = self.profile - other.profile
        return p

    def add_chirp(self, chirp: ChirpCfg):
        self._chirps.append(chirp)

    def add_frame(self, frame: FrameCfg):
        self._frames.append(frame)

    def add_profile(self, profile: ProfileCfg):
        self._profiles.append(profile)

    def describe(self) -> None:
        if self._channel:
            self._channel.describe()
        for p in self._frames:
            p.describe()
        for p in self._profiles:
            p.describe()
        for p in self._chirps:
            p.describe()

    @property
    def doppler_bins(self):
        """Create doppler bins centralized around 0"""
        return np.multiply(
            np.arange(-self.num_doppler_bins // 2 + 0.5,
                      self.num_doppler_bins // 2 + 0.5,
                      1),
            self.doppler_resolution)

    @property
    def doppler_resolution(self):
        """Calculate the doppler resolution for the given radar configuration.

        Formula: (2 * max_velocity) / number of loops

        """
        return (2 * self.max_velocity) / self._frames[0].numLoops

    def get_range_max(self, visibility_factor=0.8):
        """
            visibility_factor = 0.8 # TI specific, take SNR into account based on software version

            Returns the maximum range in meters of this config
        """
        if_max = self.profile.digOutSampleRate
        c = 299792458  # speed of light (m/s)

        beat_freq = visibility_factor * if_max
        S = self.profile.freqSlopeConst * 1e6  # slope of the transmitted chirp (MHz/s)
        range_max = (beat_freq * c) / (2 * S)

        return range_max

    def get_max_velocity(self):
        """Speed of light / (4 * chirp_repetition_time * carrier_frequency)"""
        c = 299792458  # speed of light (m/s)
        start_freq = self._profiles[0].startFreq
        freq_slope = self._profiles[0].freqSlopeConst
        adc_start_time = self._profiles[0].adcStartTime

        chirp_time = self._profiles[0].numAdcSamples / self._profiles[0].digOutSampleRate
        sweep_bandwith = freq_slope * chirp_time
        carrier_frequency = (start_freq + (freq_slope * adc_start_time * 1e-3)) + (sweep_bandwith / 2 * 1e-3)

        idle_time = self._profiles[0].idleTime
        ramp_end_time = self._profiles[0].rampEndTime
        chirp_repetition_period = self.num_tx_antennas * (idle_time + ramp_end_time)

        return c / (4 * chirp_repetition_period * carrier_frequency / 1e-3)

    @property
    def max_velocity(self):
        return self.get_max_velocity()

    @property
    def num_adc_samples(self) -> int:
        return self._profiles[0].num_adc_samples

    @property
    def num_chirps_per_frame(self) -> int:
        # this might be wrong if the chirps are not all saved in the cfg file
        return int(len(self._chirps) * self._frames[0].num_loops)

    @property
    def num_doppler_bins(self):
        # Typically the number of doppler bins is the same as the number of chirps per tx antenna
        return int(self.num_chirps_per_frame // self.num_tx_antennas)

    @property
    def num_range_bins(self):
        return self._profiles[0].num_range_bins

    @property
    def num_rx_antennas(self) -> int:
        return self._channel.num_rx_antennas

    @property
    def num_tx_antennas(self) -> int:
        # Weird, the numbers are sometimes mixed : self._channel.num_tx_antennas is just a mask)
        return len(self._chirps)  # self._channel.num_tx_antennas

    @property
    def num_virt_antennas(self) -> int:
        return self._channel.num_tx_antennas * self._channel.num_rx_antennas

    @property
    def profile(self, idx=0) -> ProfileCfg:
        return self._profiles[idx]

    @property
    def range_bins(self) -> np.array:
        """Create range_bins. Use number of adc_samples because otherwise range bins are incorrect."""
        range_max = self.get_range_max(visibility_factor=1.0)  # Same as self.range_resolution * self.num_adc_samples

        # Substract range bias if known
        # TODO: this is a serpate setting in the config
        # if 'rangeBias' in config.profile.keys():
        #     range_bias = config.profile['rangeBias']
        # else:
        range_bias = 0
        # Create evenly spaced range_bins. self.range_resolution * self.num_range_bins gives incorrect results
        # TI is consulted for clarity
        range_bins = np.linspace(0, range_max, self.num_range_bins)
        range_bins = np.subtract(range_bins, range_bias)
        range_bins = np.maximum(range_bins, 0)
        return range_bins

    @property
    def range_max(self, ) -> float:
        """
            visibility_factor = 0.8 # TI specific, take SNR into account based on software version

            Returns the maximum range in meters of this config
        """
        return self.get_range_max()

    @property
    def bandwidth(self):
        """
        Args:
            num_adc_samples (int): The number of given ADC samples in a chirp
            dig_out_sample_rate (int): The ADC sample rate
            freq_slope_const (float): The slope of the freq increase in each chirp

        Returns:
                band_width (float): The bandwidth of the radar chirp config (Hz)
        """
        freq_slope_const = self._profiles[0].freqSlopeConst  # MHz/us
        dig_out_sample_rate = self._profiles[0].digOutSampleRate  # Ms/s
        num_adc_samples = self.num_adc_samples
        freq_slope_m_hz_per_usec = freq_slope_const
        adc_sample_period_usec = 1000.0 / dig_out_sample_rate * num_adc_samples
        band_width = freq_slope_m_hz_per_usec * adc_sample_period_usec * 1e3

        return band_width

    @property
    def range_resolution(self):
        """ Calculate the range resolution for the given radar configuration

        Args:
            band_width (float): The bandwidth of the radar chirp config (Hz)

        Returns:
            float
                range_resolution (float): The range resolution for this bin (m)
        """
        c = 299792458
        range_resolution = c / (2.0 * self.bandwidth)

        return range_resolution

    def set_channel(self, channel: ChannelCfg):
        self._channel = channel

    def set_adc_cfg(self, adc_cfg: AdcCfg):
        self._adc_cfg = adc_cfg

    def set_adc_buf_cfg(self, adc_buf_cfg: AdcBufCfg):
        self._adc_buf_cfg = adc_buf_cfg

    def set_ant_geometry0(self, ant_geometry: AntGeometry):
        self._ant_geometry0 = ant_geometry

    def set_ant_geometry1(self, ant_geometry: AntGeometry):
        self._ant_geometry1 = ant_geometry

    def set_ant_phase_rot(self, ant_phase_rot: AntPhaseRot):
        self._ant_phase_rot = ant_phase_rot

    def set_clutter_removal(self, clutter_removal: ClutterRemoval):
        self._clutter_removal = clutter_removal

    def set_calib_dc_range_sig(self, calib_dc_range_sig: CalibDcRangeSig):
        self._calib_dc_range_sig = calib_dc_range_sig

    def set_comp_range_bias(self, comp_range_bias: CompRangeBiasAndRxChanPhase):
        self._comp = comp_range_bias

    def set_lvds_stream_cfg(self, lvds_stream_cfg: LvdsStreamCfg):
        self._lvds_stream_cfg = lvds_stream_cfg
