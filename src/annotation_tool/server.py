"""Serve the annotation page, camera JPEGs and CSV editing API.

GET / and /logic.js serve the UI; GET /api/recordings lists recordings and marks.
GET /frame/<recording>/<n>.jpg reads a frame in capture order.
PUT /api/recording/<recording> saves intervals and review status atomically.
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
MAX_BODY = 1_000_000  # Maximum annotation request body size in bytes.


class Catalogue:
    """Index camera frames by recording and capture position without decoding JPEGs."""

    def __init__(self, ds: Dataset) -> None:
        """Build capture-position lookups from the selected baked dataset."""
        meta = ds.select_columns(["recording", "person", "timestamp", "frame_idx"]).to_pandas()
        meta["position"] = capture_positions(meta)
        self.rows: dict[str, np.ndarray] = {}  # Recording -> dataset row at each capture position.
        for recording, frames in meta.groupby("recording"):
            rows = np.empty(len(frames), dtype=int)
            rows[frames.position.to_numpy()] = frames.index.to_numpy()
            self.rows[str(recording)] = rows
        self.person = {str(rec): str(person) for rec, person in meta.groupby("recording").person.first().items()}
        self.images: Any = ds.select_columns(["image"]).cast_column("image", HFImage(decode=False))

    @property
    def names(self) -> list[str]:
        """Recording names sorted alphabetically."""
        return sorted(self.rows)

    def frame(self, recording: str, position: int) -> bytes:
        """Return the original JPEG bytes for one capture position.

        Args:
            recording: Recording name from the catalogue.
            position: Zero-based position in capture order.

        Raises:
            KeyError: If the recording or position is unknown.
        """
        rows = self.rows[recording]
        if not 0 <= position < len(rows):
            raise KeyError(position)
        return self.images[int(rows[position])]["image"]["bytes"]


class Annotator:
    """Read and atomically update recording marks in one CSV, serialising local access."""

    def __init__(self, catalogue: Catalogue, path: Path) -> None:
        """Store the catalogue and CSV path, with a lock for this server's requests."""
        self.catalogue = catalogue
        self.path = path
        self.lock = threading.Lock()

    def state(self) -> list[dict[str, Any]]:
        """Return each recording's person, frame count, intervals and review status."""
        with self.lock:
            saved = read_annotations(self.path)
        out = []
        for name in self.catalogue.names:
            marks = saved.get(name, Marks())
            out.append({"name": name, "person": self.catalogue.person[name], "frames": len(self.catalogue.rows[name]),
                        "intervals": [list(i) for i in marks.intervals], "reviewed": marks.reviewed})
        return out

    def save(self, recording: str, intervals: Any, reviewed: Any) -> dict[str, Any]:
        """Check and save one recording's intervals and review status.

        Re-read the CSV under the lock to preserve marks for other recordings.

        Args:
            recording: Recording name offered by the catalogue.
            intervals: List of inclusive [first, last] integer capture positions.
            reviewed: Boolean indicating that the whole recording has been checked.

        Returns:
            Saved recording name, merged intervals and review status.

        Raises:
            KeyError: If the recording is not in the catalogue.
            ValueError: If values are malformed or an interval extends past the recording.
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
    """Check whether the HTTP Host header is permitted for this server's port.

    Allow loopback names and addresses. In LAN mode, also allow private IP
    addresses and .local names. Reject other domain names and mismatched ports.
    """
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
        return name.endswith(".local")  # Permit local mDNS names, e.g. my-mac.local.


def make_server(annotator: Annotator, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Create the threaded annotation server without starting its request loop.

    Args:
        annotator: Catalogue and CSV editing service.
        port: Listening port; zero lets the operating system choose a free port.
        host: Bind address; 127.0.0.1 is local only, 0.0.0.0 exposes all IPv4 interfaces.

    Returns:
        A server ready for serve_forever; the caller must close it when finished.
    """
    lan = host not in ("127.0.0.1", "localhost", "::1")

    class Handler(BaseHTTPRequestHandler):
        """Handle annotation page, frame and recording API requests."""

        server: ThreadingHTTPServer

        def log_message(self, *args: Any) -> None:
            """Suppress per-request HTTP logs so frame playback does not flood the terminal."""
            pass

        def reply(self, status: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
            """Send a byte response with its content type, length and cache policy."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def reply_json(self, status: int, value: Any) -> None:
            """Encode a value as JSON and send it with the given HTTP status."""
            self.reply(status, json.dumps(value).encode(), "application/json")

        def allowed(self) -> bool:
            """Reject requests whose Host header is outside the configured allowlist."""
            if host_allowed(self.headers.get("Host"), self.server.server_port, lan):
                return True
            self.reply_json(HTTPStatus.FORBIDDEN, {"error": "wrong Host header"})
            return False

        def do_GET(self) -> None:
            """Serve the UI, recording catalogue or a camera JPEG after checking the host."""
            if not self.allowed():
                return
            path = self.path.split("?")[0]
            if path in PAGES:
                name, content_type = PAGES[path]
                self.reply(HTTPStatus.OK, (STATIC / name).read_bytes(), content_type)
            elif path == "/api/recordings":
                try:
                    self.reply_json(HTTPStatus.OK, annotator.state())
                except ValueError as error:  # Report malformed CSV data instead of hiding it.
                    self.reply_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            elif path.startswith("/frame/") and path.endswith(".jpg"):
                try:
                    _, _, recording, number = path.split("/")
                    jpeg = annotator.catalogue.frame(unquote(recording), int(number[:-4]))
                except (KeyError, ValueError):
                    self.reply_json(HTTPStatus.NOT_FOUND, {"error": "no such frame"})
                else:
                    self.reply(HTTPStatus.OK, jpeg, "image/jpeg", cache="max-age=86400")  # Frames are immutable.
            else:
                self.reply_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_PUT(self) -> None:
            """Validate and save one recording's marks, returning a JSON result or error."""
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
            except (ValueError, AttributeError) as error:  # Invalid JSON, object shape or marks.
                self.reply_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self.reply_json(HTTPStatus.OK, result)

    return ThreadingHTTPServer((host, port), Handler)
