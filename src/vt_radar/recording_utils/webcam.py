import sys
import threading
import time

import cv2
import numpy as np
from loguru import logger


def _cv_backend() -> int:
    """Return the appropriate OpenCV capture backend for the current platform."""
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def webcam_thread(consumer, webcam):
    for img_encode in webcam.read_continuous():
         consumer(img_encode)
         time.sleep(0.01)

class Webcam(object):
    """ Uses a callback mechanism """
    def __init__(self, consumer, capture_id=1):
        self._capture = cv2.VideoCapture(capture_id, _cv_backend())
        self._error = False

        if not self._capture.isOpened():
            logger.error("Webcam on '{}' not opened", capture_id)
            self._error = True
            return

        # Minimize internal buffer so read() returns the most recent frame,
        # not a stale buffered one.  This prevents camera-radar timing drift
        # caused by the camera buffering frames while the radar serial port
        # blocks.  If the backend ignores this, set() returns False silently.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._consumer = consumer

    def read_continuous(self):
        # the error checking is done in dummy mode
        # and can be used for recording ie no image data available
        if self._error:
            while True:
                yield None
                
        while True:
            ret, frame = self._capture.read()
            if not ret:
                break
            img_encode = cv2.imencode('.jpg', frame)[1]
            yield img_encode

    def start(self):
        self._thread = threading.Thread(target=webcam_thread, args=(self._consumer, self,))
        self._thread.start()