# ----- Imports -------------------------------------------------------

# Standard Imports
import os
import struct
import serial
import time
import numpy as np
import math
import datetime
import subprocess
import re
import serial.tools.list_ports
from loguru import logger

# Local Imports
from vt_radar.parsing_utils.parseFrame import *

# Silicon Labs CP2105 dual-UART bridge used by the TI xWR/IWR EVMs:
# interface 0 = Enhanced (CLI/config), interface 1 = Standard (data).
_CP2105_VID = 0x10C4
_CP2105_PID = 0xEA70

# Port descriptions match on TI XDS110 EVMs and on CP210x under Windows/Linux. macOS
# reports neither "Enhanced" nor "Standard", so we fall back to VID/PID + interface.
_CLI_PORT_NAMES = ('XDS110 Class Application/User UART', 'Enhanced COM Port', 'Enhanced Com Port')
_DATA_PORT_NAMES = ('XDS110 Class Auxiliary Data Port', 'Standard COM Port', 'Standard Com Port')


def _port_interface(port):
    """Best-effort USB interface number of a serial port (0/1 for the CP2105), else None."""
    loc = getattr(port, 'location', None) or ''
    m = re.search(r'[.:](\d+)$', loc)
    return int(m.group(1)) if m else None


def find_com_ports():
    cli_com = None
    data_com = None
    serialPorts = list(serial.tools.list_ports.comports())

    # 1) Match by description (TI XDS110 EVMs, and CP210x on Windows/Linux).
    for port in serialPorts:
        desc = port.description or ''
        if any(name in desc for name in _CLI_PORT_NAMES):
            cli_com = port.device
        elif any(name in desc for name in _DATA_PORT_NAMES):
            data_com = port.device

    # 2) Robust fallback for the CP2105 dual-UART bridge (e.g. macOS, where descriptions are
    #    generic): identify both interfaces by VID/PID and split by interface number
    #    (0 = Enhanced/CLI, 1 = Standard/data); if the interface isn't exposed, order stably.
    if cli_com is None or data_com is None:
        cp = [p for p in serialPorts if (p.vid, p.pid) == (_CP2105_VID, _CP2105_PID)]
        if len(cp) == 2:
            cp.sort(key=lambda p: (_port_interface(p) if _port_interface(p) is not None else 0,
                                   p.location or '', p.device))
            cli_com = cli_com or cp[0].device
            data_com = data_com or cp[1].device

    if cli_com:
        print(f'\tCLI COM Port found: {cli_com}')
    if data_com:
        print(f'\tData COM Port found: {data_com}')
    logger.info("Connecting to {} and {}", cli_com, data_com)
    return cli_com, data_com
#Initialize this Class to create a UART Parser. Initialization takes one argument:
# 1: String Lab_Type - These can be:

#   a. 3D People Tracking
#   b. SDK Out of Box Demo
#   c. Long Range People Detection
#   d. Indoor False Detection Mitigation
#   e. (Legacy): Overhead People Counting
#   f. (Legacy) 2D People Counting
# Default is (f). Once initialize, call connectComPorts(self, cliComPort, DataComPort) to connect to device com ports.
# Then call readAndParseUart() to read one frame of data from the device. The gui this is packaged with calls this every frame period.
# readAndParseUart() will return all radar detection and tracking information.
class uartParser():
    def __init__(self, type='SDK Out of Box Demo'):
        # Set this option to 1 to save UART output from the radar device
        self.saveBinary = 1
        self.replay = 0
        if(self.saveBinary == 1):
            self.binData = bytearray(0)
            self.uartCounter = 0
            self.framesPerFile = 100
            self.first_file = True
            self.filepath = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            
        if (type == DEMO_NAME_OOB):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_LRPD):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_3DPC):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_SOD):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_VITALS):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_MT):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_GESTURE):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_x432_OOB):
            self.parserType = "SingleCOMPort"
        elif (type == DEMO_NAME_x432_GESTURE):
            self.parserType = "SingleCOMPort"
        # TODO Implement these
        elif (type == "Replay"):
            self.replay = 1
        else: 
            logger.error("ERROR, unsupported demo type selected!")
        
        # Data storage
        self.now_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    
    def auto_connect(self):
        cli, data = find_com_ports()
        try:
            self.connectComPorts(cli, data)
        except Exception as e:
            logger.exception(e)

    # This function is always called - first read the UART, then call a function to parse the specific demo output
    # This will return 1 frame of data. This must be called for each frame of data that is expected. It will return a dict containing all output info
    # Point Cloud and Target structure are liable to change based on the lab. Output is always cartesian.
    # DoubleCOMPort means this function refers to the xWRx843 family of devices.
    def readAndParseUartDoubleCOMPort(self):
        self.fail = 0
        if (self.replay):
            return self.replayHist()
    
        # Find magic word, and therefore the start of the frame
        index = 0
        magicByte = self.dataCom.read(1)
        frameData = bytearray(b'')
        while (1):
            # If the device doesnt transmit any data, the COMPort read function will eventually timeout
            # Which means magicByte will hold no data, and the call to magicByte[0] will produce an error
            # This check ensures we can give a meaningful error
            if (len(magicByte) < 1):
                logger.error("ERROR (timeout): No data detected on COM Port, read timed out")
                logger.error("\tBe sure that the device is in the proper mode, and that the cfg you are sending is valid")
                magicByte = self.dataCom.read(1)
                logger.error("Returning empty frame-data, please abort recording if error persists")
                return frameData
                
            # Found matching byte
            elif (magicByte[0] == UART_MAGIC_WORD[index]):
                index += 1
                frameData.append(magicByte[0])
                if (index == 8): # Found the full magic word
                    break
                magicByte = self.dataCom.read(1)
                
            else:
                # When you fail, you need to compare your byte against that byte (ie the 4th) AS WELL AS compare it to the first byte of sequence
                # Therefore, we should only read a new byte if we are sure the current byte does not match the 1st byte of the magic word sequence
                if (index == 0): 
                    magicByte = self.dataCom.read(1)
                index = 0 # Reset index
                frameData = bytearray(b'') # Reset current frame data
        
        # Read in version from the header
        versionBytes = self.dataCom.read(4)
        
        frameData += bytearray(versionBytes)

        # Read in length from header
        lengthBytes = self.dataCom.read(4)
        frameData += bytearray(lengthBytes)
        frameLength = int.from_bytes(lengthBytes, byteorder='little')
        
        # Subtract bytes that have already been read, IE magic word, version, and length
        # This ensures that we only read the part of the frame in that we are lacking
        frameLength -= 16 

        # Read in rest of the frame
        frameData += bytearray(self.dataCom.read(frameLength))
 
        # frameData now contains an entire frame, send it to parser
        return frameData

    def read_continuous(self):
        while True:
            yield self.readAndParseUartDoubleCOMPort()

    # This function is identical to the readAndParseUartDoubleCOMPort function, but it's modified to work for SingleCOMPort devices in the xWRLx432 family
    def readAndParseUartSingleCOMPort(self):
        # Reopen CLI port
        if not self.cliCom.isOpen():
            logger.error("Reopening Port")
            self.cliCom.open()

        self.fail = 0
        # Find magic word, and therefore the start of the frame
        index = 0
        magicByte = self.cliCom.read(1)
        frameData = bytearray(b'')
        while (1):
            # If the device doesnt transmit any data, the COMPort read function will eventually timeout
            # Which means magicByte will hold no data, and the call to magicByte[0] will produce an error
            # This check ensures we can give a meaningful error
            if (len(magicByte) < 1):
                print ("ERROR: No data detected on COM Port, read timed out")
                print("\tBe sure that the device is in the proper mode, and that the cfg you are sending is valid")
                magicByte = self.cliCom.read(1)

            # Found matching byte
            elif (magicByte[0] == UART_MAGIC_WORD[index]):
                index += 1
                frameData.append(magicByte[0])
                if (index == 8): # Found the full magic word
                    break
                magicByte = self.cliCom.read(1)
                
            else:
                # When you fail, you need to compare your byte against that byte (ie the 4th) AS WELL AS compare it to the first byte of sequence
                # Therefore, we should only read a new byte if we are sure the current byte does not match the 1st byte of the magic word sequence
                if (index == 0):
                    magicByte = self.cliCom.read(1)
                index = 0 # Reset index
                frameData = bytearray(b'') # Reset current frame data
        
        # Read in version from the header
        versionBytes = self.cliCom.read(4)
        
        frameData += bytearray(versionBytes)

        # Read in length from header
        lengthBytes = self.cliCom.read(4)
        frameData += bytearray(lengthBytes)
        frameLength = int.from_bytes(lengthBytes, byteorder='little')
        
        # Subtract bytes that have already been read, IE magic word, version, and length
        # This ensures that we only read the part of the frame in that we are lacking
        frameLength -= 16 

        # Read in rest of the frame
        frameData += bytearray(self.cliCom.read(frameLength))
 
        # frameData now contains an entire frame, send it to parser
        if (self.parserType == "SingleCOMPort"):
            raw_data = parseStandardFrame(frameData)
            outputDict = raw_data.to_dict()
        else:
            logger.error('FAILURE: Bad parserType')
            return None
        
        return outputDict

    # Used on raspberry pi: Power cycle the radar device before starting
    def powerCycleRadar(self):
        # On raspberry pi, power cycle the USB before connecting
        if os.name != 'nt' and "raspberrypi" in os.uname().nodename.lower():
            subprocess.run(
                "uhubctl -l 1-1 -a cycle > /dev/null 2>&1",
                shell=True
            )
            time.sleep(2)

    # Find various utility functions here for connecting to COM Ports, send data, etc...
    # Connect to com ports
    # Call this function to connect to the comport. This takes arguments self (intrinsic), cliCom, and dataCom. No return, but sets internal variables in the parser object.
    def connectComPorts(self, cliCom, dataCom):
        self.powerCycleRadar()
        self.cliCom = serial.Serial(cliCom, 115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom = serial.Serial(dataCom, 921600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom.reset_output_buffer()
        logger.info('Connected {} {}', self.cliCom, self.dataCom)

    # Separate connectComPort (not PortS) for IWRL6432 because it only uses one port
    def connectComPort(self, cliCom, cliBaud=115200):
        # Longer timeout time for IWRL6432 to support applications with low power / low update rate
        self.powerCycleRadar()
        self.dataCom = serial.Serial(cliCom, cliBaud, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=4)
        self.dataCom.reset_output_buffer()
        logger.info('Connected (one port)')

    #send cfg over uart
    def sendCfg(self, cfg):
        # Ensure each line ends in \n for proper parsing
        cfg = _normalize_cfg_lines(cfg)

        for line in cfg:
            time.sleep(.03) # Line delay

            if(self.cliCom.baudrate == 1250000):
                for char in [*line]:
                    time.sleep(.001) # Character delay. Required for demos which are 1250000 baud by default else characters are skipped
                    self.cliCom.write(char.encode())
            else:
                self.cliCom.write(line.encode())
                
            ack = self.cliCom.readline()
            #print(ack)
            ack = self.cliCom.readline()
            #print(ack)
            splitLine = line.split()
            if(splitLine[0] == "baudRate"): # The baudrate CLI line changes the CLI baud rate on the next cfg line to enable greater data streaming off the IWRL device.
                try:
                    self.cliCom.baudrate = int(splitLine[1])
                except:
                    logger.exception("Error - Invalid baud rate")
                    sys.exit(1)
        # Give a short amount of time for the buffer to clear
        time.sleep(0.03)
        self.cliCom.reset_input_buffer()
        # NOTE - Do NOT close the CLI port because 6432 will use it after configuration


    #send single command to device over UART Com.
    def sendLine(self, line):
        self.cliCom.readline()
        self.cliCom.readline()
        self.cliCom.readline()
        self.cliCom.write(line.encode())
        ack = self.cliCom.readline()
        print(ack)
        ack = self.cliCom.readline()
        print(ack)

    # def replayHist(self):
    #     if (self.replayData):
    #         #print('reading data')
    #         #print('fail: ',self.fail)
    #         #print(len(self.replayData))
    #         #print(self.replayData[0:8])
    #         self.replayData = self.Capon3DHeader(self.replayData)
    #         #print('fail: ',self.fail)
    #         return self.pcBufPing, self.targetBufPing, self.indexes, self.numDetectedObj, self.numDetectedTarget, self.frameNum, self.fail, self.classifierOutput
    #         #frameData = self.replayData[0]
    #         #self.replayData = self.replayData[1:]
    #         #return frameData['PointCloud'], frameData['Targets'], frameData['Indexes'], frameData['Number Points'], frameData['NumberTracks'],frameData['frame'],0, frameData['ClassifierOutput'], frameData['Uniqueness']
    #     else:
    #         filename = 'overheadDebug/binData/pHistBytes_'+str(self.saveNum)+'.bin'
    #         #filename = 'Replay1Person10mShort/pHistRT'+str(self.saveNum)+'.pkl'
    #         self.saveNum+=1
    #         try:
    #             dfile = open(filename, 'rb', 0)
    #         except:
    #             print('cant open ', filename)
    #             return -1
    #         self.replayData = bytes(list(dfile.read()))
    #         if (self.replayData):
    #             print('entering replay')
    #             return self.replayHist()
    #         else:
    #             return -1
        
def _normalize_cfg_lines(cfg):
    """Drop blank lines and ensure every remaining line ends with '\\n'.

    Builds a new list instead of mutating ``cfg`` in place. The old in-place
    version removed items while iterating with ``enumerate``, so the element
    that shifted into a removed slot was skipped: consecutive blank lines left
    a stray ``'\\n'`` behind, which later crashed the send loop
    (``line.split()[0]`` raises IndexError on an empty line).
    """
    normalized = []
    for line in cfg:
        # Remove empty lines from cfg
        if line == '\n':
            continue
        # add a newline to the end of every line (protects against the last
        # line not having a newline at the end of it)
        if not line.endswith('\n'):
            line = line + '\n'
        normalized.append(line)
    return normalized


def getBit(byte, bitNum):
    mask = 1 << bitNum
    if (byte&mask):
        return 1
    else:
        return 0