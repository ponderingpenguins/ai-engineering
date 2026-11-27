import math
import sys
import os
import pandas as pd
from vt_radar.mmwave.dataloader.adc import DCA1000
from vt_radar.recording_utils.webcam import Webcam
from datetime import datetime
from loguru import logger
from halo import Halo
from datetime import datetime
import shutil

def do_record(cfg_file):
    logger.info("Start recording usb camera frames")
    dca = DCA1000()
    webcam = Webcam(None)
    rows = []
    FRAMES_PER_FILE = 100

    directory = "recording"
    spinner = Halo(text='Recording... ', spinner='bouncingBar', color='green', text_color='cyan')
    spinner.start()
    
    # Get the current date and time in a sortable format
    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
    recording_dir = os.path.join(directory, formatted_datetime)

    if not os.path.exists(recording_dir):
        # Create the directory
        os.makedirs(recording_dir)

    logger.info(f"Recording to destination: {recording_dir}")

   # config_filename = os.path.basename(cfg_file)
   # destination = os.path.join(recording_dir, config_filename)
  #  shutil.copy2(cfg_file, destination)

    for idx, (im, frame) in enumerate(zip(webcam.read_continuous(), dca.read_continuous())):
        adc = DCA1000.organize(raw_frame=frame, num_chirps=384,
                               num_rx=4, num_samples=128)
        print(adc)
        row = {"mmwave": adc, "timestamp" : datetime.utcnow()}
        if im is not None:
            row["image"] = im
        rows.append(row)
        if len(rows) == FRAMES_PER_FILE:
            df = pd.DataFrame(rows)
            filename = 'data_' + "%04d" % math.floor(idx/FRAMES_PER_FILE)+'.h5'
            fn = os.path.join(recording_dir, filename)
            print(df.describe())
            df.to_hdf(fn, key='df', mode='w')
            rows = []
    print(df)
    print("Done")
    spinner.stop()


if __name__ == "__main__":
    do_record(None)