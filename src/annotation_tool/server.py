"""The local server of the annotation tool: the page, the camera frames and the saving of the marks.

It uses only the standard library and listens on the local machine only. The page is plain HTML and JS.

    GET  /                           the page (and /logic.js)
    GET  /api/recordings             every recording with its marks, as saved in the file
    GET  /frame/<recording>/<n>.jpg  the original camera frame at capture position n
    PUT  /api/recording/<recording>  {"intervals": [[first, last], ...], "reviewed": bool}: saves the marks
"""

import ipaddress
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
from datasets import Dataset
from datasets import Image as HFImage

from data.annotations import Marks, capture_positions, make_marks, read_annotations, write_annotations

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
PAGES = {"/": ("index.html", "text/html; charset=utf-8"), "/logic.js": ("logic.js", "text/javascript; charset=utf-8")}
MAX_BODY = 1_000_000  # bytes; a recording has a few hundred intervals at most


class Catalogue:
    """The recordings of a dataset and their frames in capture order. Reads the original JPEGs."""

    def __init__(self, ds: Dataset) -> None:
        meta = ds.select_columns(["recording", "person", "timestamp", "frame_idx"]).to_pandas()
        meta["position"] = capture_positions(meta)
        self.rows: dict[str, np.ndarray] = {}  # recording -> the dataset row of every capture position
        for recording, frames in meta.groupby("recording"):
            rows = np.empty(len(frames), dtype=int)
            rows[frames.position.to_numpy()] = frames.index.to_numpy()
            self.rows[str(recording)] = rows
        self.person = {str(rec): str(person) for rec, person in meta.groupby("recording").person.first().items()}
        self.images: Any = ds.select_columns(["image"]).cast_column("image", HFImage(decode=False))

    @property
    def names(self) -> list[str]:
        """The recordings in the order of their names, which is the order of recording time."""
        return sorted(self.rows)

    def frame(self, recording: str, position: int) -> bytes:
        """The JPEG bytes of one frame.

        Raises:
            KeyError: If the recording or the position does not exist.
        """
        rows = self.rows[recording]
        if not 0 <= position < len(rows):
            raise KeyError(position)
        return self.images[int(rows[position])]["image"]["bytes"]


class Annotator:
    """The marks of the recordings in one file. Every save reads the file, changes one recording and writes it back."""

    def __init__(self, catalogue: Catalogue, path: Path) -> None:
        self.catalogue = catalogue
        self.path = path
        self.lock = threading.Lock()

    def state(self) -> list[dict[str, Any]]:
        """Every recording with its person, its number of frames and its saved marks."""
        with self.lock:
            saved = read_annotations(self.path)
        out = []
        for name in self.catalogue.names:
            marks = saved.get(name, Marks())
            out.append({"name": name, "person": self.catalogue.person[name], "frames": len(self.catalogue.rows[name]),
                        "intervals": [list(i) for i in marks.intervals], "reviewed": marks.reviewed})
        return out

    def save(self, recording: str, intervals: Any, reviewed: Any) -> dict[str, Any]:
        """Replaces the marks of one recording.

        Raises:
            KeyError: If the recording is not offered by the tool.
            ValueError: If the input is not valid, or an interval reaches past the end of the recording.
        """
        if recording not in self.catalogue.rows:
            raise KeyError(recording)
        if not isinstance(reviewed, bool) or not isinstance(intervals, list):
            raise ValueError('expected {"intervals": [[first, last], ...], "reviewed": true or false}')
        if not all(isinstance(i, list) and len(i) == 2 and all(isinstance(p, int) and not isinstance(p, bool) for p in i)
                   for i in intervals):
            raise ValueError("every interval must be a list of two integers")
        marks = make_marks(intervals, reviewed)
        frames = len(self.catalogue.rows[recording])
        if marks.intervals and marks.intervals[-1][1] >= frames:
            raise ValueError(f"{recording} has only {frames} frames, the last position is {frames - 1}")
        with self.lock:
            saved = read_annotations(self.path)
            saved[recording] = marks
            write_annotations(saved, self.path)
        logger.info("saved %s: %d intervals, %s", recording, len(marks.intervals),
                    "reviewed" if marks.reviewed else "not reviewed")
        return {"name": recording, "intervals": [list(i) for i in marks.intervals], "reviewed": marks.reviewed}


def host_allowed(header: str | None, port: int, lan: bool) -> bool:
    """Is `Host` a name for this machine? On the LAN only an IP address of a private network is accepted, not a
    domain name, so a web page cannot rebind its own domain name to us and read the frames in the browser."""
    if not header:
        return False
    try:
        parts = urlsplit("//" + header)
        name, given_port = parts.hostname, parts.port
    except ValueError:
        return False
    if name is None or given_port != port:
        return False
    if name in ("127.0.0.1", "localhost", "::1"):
        return True
    if not lan:
        return False
    try:
        return ipaddress.ip_address(name).is_private
    except ValueError:
        return name.endswith(".local")  # mDNS name of the machine, e.g. my-mac.local


def make_server(annotator: Annotator, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """A server on the local machine, or on every network interface when `host` is `0.0.0.0`.
    Use `port=0` to let the system pick a free port."""
    lan = host not in ("127.0.0.1", "localhost", "::1")

    class Handler(BaseHTTPRequestHandler):
        server: ThreadingHTTPServer

        def log_message(self, *args: Any) -> None:  # the frames would flood the terminal
            pass

        def reply(self, status: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def reply_json(self, status: int, value: Any) -> None:
            self.reply(status, json.dumps(value).encode(), "application/json")

        def allowed(self) -> bool:
            """Only requests addressed to this machine: it blocks other sites from using the browser to reach us."""
            if host_allowed(self.headers.get("Host"), self.server.server_port, lan):
                return True
            self.reply_json(HTTPStatus.FORBIDDEN, {"error": "wrong Host header"})
            return False

        def do_GET(self) -> None:
            if not self.allowed():
                return
            path = self.path.split("?")[0]
            if path in PAGES:
                name, content_type = PAGES[path]
                self.reply(HTTPStatus.OK, (STATIC / name).read_bytes(), content_type)
            elif path == "/api/recordings":
                try:
                    self.reply_json(HTTPStatus.OK, annotator.state())
                except ValueError as error:  # the file is damaged: say so instead of hiding it
                    self.reply_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            elif path.startswith("/frame/") and path.endswith(".jpg"):
                try:
                    _, _, recording, number = path.split("/")
                    jpeg = annotator.catalogue.frame(unquote(recording), int(number[:-4]))
                except (KeyError, ValueError):
                    self.reply_json(HTTPStatus.NOT_FOUND, {"error": "no such frame"})
                else:
                    self.reply(HTTPStatus.OK, jpeg, "image/jpeg", cache="max-age=86400")  # a frame never changes
            else:
                self.reply_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_PUT(self) -> None:
            if not self.allowed():
                return
            prefix = "/api/recording/"
            if not self.path.startswith(prefix):
                self.reply_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_BODY:
                self.reply_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too large"})
                return
            try:
                body = json.loads(self.rfile.read(size))
                result = annotator.save(unquote(self.path[len(prefix):]), body.get("intervals"), body.get("reviewed"))
            except KeyError:
                self.reply_json(HTTPStatus.NOT_FOUND, {"error": "no such recording"})
            except (ValueError, AttributeError) as error:  # bad JSON, a body that is not an object, or invalid marks
                self.reply_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self.reply_json(HTTPStatus.OK, result)

    return ThreadingHTTPServer((host, port), Handler)
