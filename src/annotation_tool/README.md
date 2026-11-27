# Lying Annotation Tool (`src/annotation_tool/`)

A self-contained local web application (Python standard library server and vanilla HTML/JavaScript frontend) for reviewing top-down camera frames in capture order and saving per-recording lying intervals directly to `data/annotations/lying_intervals.csv`.

## Modules

| File                  | Responsibility                                                                                                                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__main__.py`       | CLI entry point (`python main.py annotate`); loads `DataConfig`, filters recordings via `select_recordings`, binds the server, and opens the browser.                                           |
| `server.py`         | `Catalogue` (capture-ordered raw JPEG access), `Annotator` (thread-safe CSV state management), `make_server` (HTTP handler), and `host_allowed` (`Host` header validation).                 |
| `static/index.html` | Single-page viewer with 4:3 frame rendering, clickable/draggable timeline canvas, 40-frame preloader, serialized `PUT` save queue, and keyboard/touch bindings.                                     |
| `static/logic.js`   | Pure, DOM-free editing logic (`toIntervals`, `fromIntervals`, `History`, `mark`, `finish`, `toggleFrame`, `removeInterval`, `toggleReviewed`), runnable under Node.js for unit tests. |



## Walkthrough

1. **Startup:** `__main__.py` loads `config/data.yaml` (plus any CLI overrides), opens the baked dataset, and runs `select_recordings()` so only valid recordings (known participant, not excluded, matching sensor config) are offered. Heatmap denoising is skipped because the viewer only reads camera JPEGs.
2. **Frame ordering:** `Catalogue` indexes each recording using `data.annotations.capture_positions()`, ensuring frames are served in capture-time order `0..N-1` regardless of `.h5` file order. Images are read lazily without decoding via `HFImage(decode=False)`.
3. **Editing and saving:** Every edit mutates an immutable `Uint8Array` of frame flags in `logic.js`, pushes the previous state onto a 50-step undo stack, and queues a `PUT /api/recording/<recording>` request. Requests are chained sequentially so the last edit always wins. On the server, `Annotator.save()` merges overlapping or adjacent intervals via `data.annotations.make_marks()`, verifies bounds against the recording length, and writes the CSV via temporary-file replacement.
4. **Reviewed status:** Marking a recording as reviewed (`r`) tells `data/annotations.py` to label all frames outside the marked intervals as `"not_lying"` (a reviewed recording with no lying writes a single empty-interval row `recording,,,true`). Unreviewed recordings still save their partial intervals to CSV, but produce `lying_annotation = None` during baking so incomplete work is never treated as negative examples.
5. **Syncing into training data:** After annotating, run `python main.py bake --refresh-annotations` to write the updated `lying_annotation` column into the baked dataset, then optionally run `python main.py audit` to inspect disagreements against `label_pose` (see [src/evaluation/README.md](../evaluation/README.md)).

### Labelling Criteria

Mark a frame as **lying** when the participant's torso is on the floor and roughly horizontal (supine, prone, or on the side). The first lying frame is the first frame with the torso at floor level; the last is the final frame before the torso begins lifting. Sitting, kneeling, crawling, and leaning on the hands are not lying.

### Keyboard Controls

| Key                                    | Action                              | Key                        | Action                                 |
| -------------------------------------- | ----------------------------------- | -------------------------- | -------------------------------------- |
| `Space`                              | Play / pause (`0.5x`–`4x`)     | `i`                      | Set first lying frame                  |
| `Left` / `Right`                   | Step 1 frame (`Shift`: 10 frames) | `o`                      | Set last lying frame and fill interval |
| `Up` / `Down`                      | Previous / next recording           | `x`                      | Toggle lying flag on current frame     |
| `n`                                  | Jump to next unreviewed recording   | `Delete` / `Backspace` | Remove interval at current position    |
| `Ctrl/Cmd+Z`                         | Undo                                | `r`                      | Toggle `reviewed` status             |
| `Ctrl/Cmd+Shift+Z` or `Ctrl/Cmd+Y` | Redo                                | Timeline drag              | Scrub directly to any capture position |

Deep links jump directly to a recording and frame: `/#rec=<recording>&f=<position>` (append `&play=1` to auto-play).

### HTTP Endpoints and Access Control

| Endpoint                           | Behavior                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `GET /`, `GET /logic.js`       | Serves static UI files (`Cache-Control: no-store`).                                                   |
| `GET /api/recordings`            | Returns JSON list of recordings with `name`, `person`, `frames`, `intervals`, and `reviewed`. |
| `GET /frame/<recording>/<n>.jpg` | Serves raw JPEG bytes for capture position `n` (`Cache-Control: max-age=86400`).                    |
| `PUT /api/recording/<recording>` | Accepts `{"intervals": [[first, last], ...], "reviewed": bool}` (max 1 MB) and saves to CSV.          |

`host_allowed()` validates the `Host` header on every request to prevent DNS rebinding from other browser tabs. By default the server binds to `127.0.0.1` and accepts only loopback hostnames. Passing `--lan` binds to `0.0.0.0` and additionally allows private-network IPs and `.local` mDNS names; because camera frames show identifiable people, use `--lan` only on a trusted network.



## Usage

```bash
# Local browser session (default port 8765)
python main.py annotate

# Headless or phone access on a trusted LAN, custom CSV path
python main.py annotate --port 8765 --lan --no-browser --csv data/annotations/lying_intervals.csv

# Apply updated annotations to the baked dataset
python main.py bake --refresh-annotations
```



## Extending

1. **Adding an editing command:** Implement a pure `(state, pos) -> {state, changed, message}` function in `static/logic.js`, add Node unit tests in `tests/js/annotation_logic_test.js` (invoked by `tests/test_annotation_logic.py`), and bind the control in `static/index.html`.
2. **Adding a server route or validation rule:** Update `Handler` or `Annotator` in `server.py` and add coverage in `tests/test_annotation_server.py`.
