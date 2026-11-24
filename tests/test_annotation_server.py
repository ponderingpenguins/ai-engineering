import http.client
import io
import json
import threading
from datetime import datetime, timedelta

import pytest
from datasets import Dataset
from datasets import Image as HFImage
from PIL import Image

from annotation_tool.server import Annotator, Catalogue, host_allowed, make_server
from data.annotation import make_marks, read_annotations, write_annotations

START = datetime(2026, 1, 1)


def jpeg(red: int) -> dict:
    """A solid image whose red value tells which frame it is."""
    out = io.BytesIO()
    Image.new("RGB", (16, 12), (red, 0, 0)).save(out, "JPEG", quality=95)
    return {"bytes": out.getvalue(), "path": None}


def dataset() -> Dataset:
    """Recording "a" has 3 frames. Recording "b" has 6, stored out of time order: frame_idx 3-5 come second but
    were captured between frame_idx 0 and 1. The red value of a frame is 40 * frame_idx."""
    rows = [("a", "Bob", i, i, 40 * i) for i in range(3)]
    capture_order = [0, 3, 4, 5, 1, 2]  # frame_idx of the frames of "b" in the order they were captured
    rows += [("b", "Igor", idx, n, 40 * idx) for n, idx in enumerate(capture_order)]
    ds = Dataset.from_dict({
        "recording": [r[0] for r in rows], "person": [r[1] for r in rows], "frame_idx": [r[2] for r in rows],
        "timestamp": [START + timedelta(seconds=0.2 * r[3]) for r in rows],
        "image": [jpeg(r[4]) for r in rows],
    })
    return ds.shuffle(seed=0).flatten_indices()  # the rows are not in capture order either


@pytest.fixture
def tool(tmp_path):
    """A running server on a free port, with its marks file."""
    path = tmp_path / "lying.csv"
    annotator = Annotator(Catalogue(dataset().cast_column("image", HFImage())), path)
    server = make_server(annotator, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_port, path
    server.shutdown()
    server.server_close()


def request(port, method, url, body=None, headers=None, host=None):
    connection = http.client.HTTPConnection("127.0.0.1", port)
    connection.putrequest(method, url, skip_host=True)
    connection.putheader("Host", host or f"127.0.0.1:{port}")
    for key, value in (headers or {}).items():
        connection.putheader(key, value)
    data = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
    if data is not None:
        connection.putheader("Content-Length", str(len(data)))
    connection.endheaders(data)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, response.headers, payload


def declared_size(port, size):
    """The status of a PUT that announces `size` bytes of body but sends none."""
    connection = http.client.HTTPConnection("127.0.0.1", port)
    connection.putrequest("PUT", "/api/recording/a", skip_host=True)
    connection.putheader("Host", f"127.0.0.1:{port}")
    connection.putheader("Content-Length", str(size))
    connection.endheaders()
    status = connection.getresponse().status
    connection.close()
    return status


def get_json(port, url):
    status, _, payload = request(port, "GET", url)
    return status, json.loads(payload)


# ---- reading ---------------------------------------------------------------------------------------

def test_the_recordings_are_listed_by_name_with_their_marks(tool):
    port, path = tool
    write_annotations({"b": make_marks([(1, 2)], True)}, path)
    status, recordings = get_json(port, "/api/recordings")
    assert status == 200
    assert recordings == [
        {"name": "a", "person": "Bob", "frames": 3, "intervals": [], "reviewed": False},
        {"name": "b", "person": "Igor", "frames": 6, "intervals": [[1, 2]], "reviewed": True}]


def red_of(payload):
    return Image.open(io.BytesIO(payload)).convert("RGB").getpixel((8, 6))[0]


def test_frames_are_served_in_capture_order_not_in_frame_idx_order(tool):
    port, _ = tool
    reds = []
    for position in range(6):
        status, headers, payload = request(port, "GET", f"/frame/b/{position}.jpg")
        assert status == 200 and headers["Content-Type"] == "image/jpeg"
        reds.append(red_of(payload))
    expected = [40 * idx for idx in [0, 3, 4, 5, 1, 2]]  # frame_idx 0, 3, 4, 5, 1, 2
    assert all(abs(got - want) < 6 for got, want in zip(reds, expected)), (reds, expected)


def test_the_original_jpeg_bytes_are_served_unchanged(tool):
    port, _ = tool
    _, _, payload = request(port, "GET", "/frame/a/1.jpg")
    assert payload == jpeg(40)["bytes"]


@pytest.mark.parametrize("url", ["/frame/b/6.jpg", "/frame/b/-1.jpg", "/frame/nope/0.jpg", "/frame/b/x.jpg", "/other"])
def test_missing_frames_and_pages_are_404(tool, url):
    assert request(tool[0], "GET", url)[0] == 404


def test_the_page_and_its_script_are_served(tool):
    port, _ = tool
    status, headers, payload = request(port, "GET", "/")
    assert status == 200 and b"Lying annotation" in payload and headers["Content-Type"].startswith("text/html")
    status, headers, payload = request(port, "GET", "/logic.js")
    assert status == 200 and b"Annotation" in payload and "javascript" in headers["Content-Type"]


# ---- saving ----------------------------------------------------------------------------------------

def test_saving_writes_the_file_and_merges_the_intervals(tool):
    port, path = tool
    status, _, payload = request(port, "PUT", "/api/recording/b", {"intervals": [[4, 5], [1, 2], [3, 3]], "reviewed": True})
    assert status == 200 and json.loads(payload) == {"name": "b", "intervals": [[1, 5]], "reviewed": True}
    assert read_annotations(path) == {"b": make_marks([(1, 5)], True)}
    assert get_json(port, "/api/recordings")[1][1]["intervals"] == [[1, 5]]


def test_a_save_changes_one_recording_and_keeps_the_others_even_if_the_file_changed_meanwhile(tool):
    port, path = tool
    request(port, "PUT", "/api/recording/a", {"intervals": [[0, 0]], "reviewed": False})
    write_annotations({**read_annotations(path), "other": make_marks([], True)}, path)  # edited by hand
    request(port, "PUT", "/api/recording/b", {"intervals": [], "reviewed": True})
    saved = read_annotations(path)
    assert saved["a"] == make_marks([(0, 0)], False) and saved["b"] == make_marks([], True)
    assert saved["other"] == make_marks([], True)


def test_marks_of_an_unfinished_recording_are_saved_but_not_reviewed(tool):
    port, path = tool
    request(port, "PUT", "/api/recording/a", {"intervals": [[1, 1]], "reviewed": False})
    assert path.read_text().splitlines()[1:] == ["a,1,1,false"]


@pytest.mark.parametrize("body", [
    {"intervals": [[2, 1]], "reviewed": True},       # ends before it starts
    {"intervals": [[-1, 1]], "reviewed": True},      # negative
    {"intervals": [[0, 3]], "reviewed": True},       # recording "a" has only 3 frames: the last position is 2
    {"intervals": [[0, 1.5]], "reviewed": True},     # not an integer
    {"intervals": [[0, True]], "reviewed": True},    # a bool is not a position
    {"intervals": [[0]], "reviewed": True},          # not a pair
    {"intervals": "x", "reviewed": True},
    {"intervals": [], "reviewed": "yes"},            # reviewed must be a bool
    {"intervals": []},
    [],                                              # not an object
])
def test_invalid_saves_are_refused_and_change_nothing(tool, body):
    port, path = tool
    status, _, payload = request(port, "PUT", "/api/recording/a", body)
    assert status == 400 and "error" in json.loads(payload)
    assert not path.exists()


def test_bad_json_unknown_recordings_and_big_bodies_are_refused(tool):
    port, path = tool
    assert request(port, "PUT", "/api/recording/a", b"{not json")[0] == 400
    assert request(port, "PUT", "/api/recording/zzz", {"intervals": [], "reviewed": True})[0] == 404
    assert declared_size(port, 2_000_000) == 413  # refused from the header, before the body is read
    assert request(port, "PUT", "/elsewhere", {"intervals": [], "reviewed": True})[0] == 404
    assert not path.exists()


def test_requests_for_another_host_are_refused(tool):
    port, path = tool
    assert request(port, "GET", "/api/recordings", host="evil.example")[0] == 403
    assert request(port, "PUT", "/api/recording/a", {"intervals": [], "reviewed": True}, host="evil.example")[0] == 403
    assert request(port, "GET", "/api/recordings", host=f"localhost:{port}")[0] == 200
    assert not path.exists()


def test_a_damaged_file_is_reported_not_hidden(tool):
    port, path = tool
    path.write_text("recording,first_capture_idx,last_capture_idx,reviewed\na,3,,true\n")
    status, payload = get_json(port, "/api/recordings")
    assert status == 500 and "only one position" in payload["error"]


def test_two_saves_at_the_same_time_both_arrive(tool):
    port, path = tool
    results = []

    def save(name):
        results.append(request(port, "PUT", f"/api/recording/{name}", {"intervals": [[0, 1]], "reviewed": True})[0])

    threads = [threading.Thread(target=save, args=(name,)) for name in "ab" * 5]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200] * 10
    assert set(read_annotations(path)) == {"a", "b"}


@pytest.mark.parametrize("header, lan, expected", [
    ("127.0.0.1:8765", False, True),
    ("localhost:8765", False, True),
    ("192.168.1.20:8765", False, False),
    ("192.168.1.20:8765", True, True),
    ("10.0.0.5:8765", True, True),
    ("my-mac.local:8765", True, True),
    ("8.8.8.8:8765", True, False),
    ("evil.example:8765", True, False),
    ("192.168.1.20:9999", True, False),
    ("192.168.1.20", True, False),
    (None, True, False),
])
def test_host_allowed(header, lan, expected):
    assert host_allowed(header, 8765, lan) is expected
