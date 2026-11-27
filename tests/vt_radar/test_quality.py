from dataclasses import replace

import numpy as np
import pytest

from vt_radar import quality as q

CFG = ["% comment", "sensorStop", "frameCfg 0 2 16 0 200 1 0", "sensorStart"]
R, D = 64, 16


@pytest.fixture
def policy():
    return q.load_policy()


@pytest.fixture
def ctx(policy):
    return q.build_context(CFG, np.linspace(0, 6, R), D, policy.roi_range_m)


def empty_frames(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(1600 + rng.normal(0, 20, (n, R, D)), 0, 65535).astype(np.uint16)


def occupied_points(n=3, snr=20.0):
    return np.array([[0.1, 1.8, 0.2, 0.0, snr, 30.0, 255.0]] * n)


def run_window(policy, ctx, ref, heatmap, points, occupancy="present", snr_available=True):
    frames = [q.measure_frame(heatmap, points, ctx, ref, policy, i, occupancy, snr_available=snr_available)
              for i in range(policy.window_len)]
    return q.score_window(frames, policy)


def test_worked_example_from_research_doc():
    doc_thresholds = {"visibility_min": 4.0, "gradient_min": 1.0, "roi_points_min": 32,
                      "empty_frame_fraction_max": 0.10, "snr_min_db": 10.0}
    policy = replace(q.load_policy("visibility_focus_v1"), thresholds=doc_thresholds)
    m = {"visibility_median": 3.0, "gradient_detail": 0.5, "roi_points": 40, "all_points": 48,
         "empty_roi_frame_fraction": 0.25, "median_point_snr_db": 8.0, "point_overflow_fraction": 0.0,
         "points_ok": True}
    p, _ = q.window_penalties(m, policy)
    assert p == pytest.approx({"visibility": 0.0625, "focus": 0.25, "point_count": 0, "empty_frames": 1,
                               "point_snr": 0.04, "point_cap": 0})
    e_heat, _ = q.combine(p, policy.weights["heatmap"])
    e_pts, _ = q.combine(p, policy.weights["pointcloud"])
    e_all, _ = q.combine({"heatmap": e_heat, "pointcloud": e_pts}, policy.weights["combined"])
    assert [round(q.score(e), 2) for e in (e_heat, e_pts, e_all)] == [90.48, 77.26, 84.94]


def test_person_raises_visibility_and_background_does_not(policy, ctx):
    ref = q.fit_empty_reference(empty_frames(), ctx, policy)
    background = ref.background.astype(np.uint16)
    person = background.copy()
    person[30:34, 0:3] += 400  # still body near zero Doppler
    empty = q.measure_frame(background, np.zeros((0, 7)), ctx, ref, policy, 0, "present")
    occupied = q.measure_frame(person, np.zeros((0, 7)), ctx, ref, policy, 0, "present")
    assert empty.visibility < policy.thresholds["visibility_min"] < occupied.visibility


def test_values_below_background_do_not_wrap_around(policy, ctx):
    ref = q.fit_empty_reference(empty_frames(), ctx, policy)
    darker = (ref.background - 300).astype(np.uint16)
    m = q.measure_frame(darker, np.zeros((0, 7)), ctx, ref, policy, 0, "present")
    assert m.visibility == 0
    assert m.changed_fraction == 1  # decreases count as change


def test_absent_signal_is_not_called_focused(policy):
    assert q.normalised_entropy(np.zeros(100), policy.min_excess) is None
    assert q.gini(np.zeros(100), policy.min_excess) is None
    spike = np.zeros(100)
    spike[7] = 50
    assert q.normalised_entropy(spike, policy.min_excess) == 0
    assert q.gini(spike, policy.min_excess) == pytest.approx(1 - 1 / 100)


def test_isolated_peak_looks_concentrated(policy, ctx):
    # Failure case from the doc: a strong reflector is focused but is not a body.
    ref = q.fit_empty_reference(empty_frames(), ctx, policy)
    peak = ref.background.astype(np.uint16)
    peak[40, 5] += 2000
    body = ref.background.astype(np.uint16)
    body[28:36, 0:4] += 250
    m_peak = q.measure_frame(peak, np.zeros((0, 7)), ctx, ref, policy, 0, "present")
    m_body = q.measure_frame(body, np.zeros((0, 7)), ctx, ref, policy, 0, "present")
    assert m_peak.gini > m_body.gini


def test_scores_need_present_occupancy_in_every_frame(policy, ctx):
    ref = q.fit_empty_reference(empty_frames(), ctx, policy)
    hm = ref.background.astype(np.uint16)
    frames = [q.measure_frame(hm, occupied_points(), ctx, ref, policy, i, "present" if i % 2 else "empty")
              for i in range(policy.window_len)]
    w = q.score_window(frames, policy)
    assert w.occupancy == "mixed"
    assert (w.heatmap_score, w.pointcloud_score, w.combined_score) == (None, None, None)

    w = run_window(policy, ctx, ref, hm, occupied_points(), occupancy="unknown")
    assert w.heatmap_score is None and w.reasons["heatmap"] == "occupancy_unknown"
    assert "weak change from reference; occupancy unknown" in w.warnings


def test_missing_reference_leaves_heatmap_unscored_but_points_scored(policy, ctx):
    w = run_window(policy, ctx, None, empty_frames(1)[0], occupied_points())
    assert w.heatmap_score is None and w.reasons["heatmap"] == "reference_missing"
    assert w.combined_score is None
    assert w.pointcloud_score is not None


def test_point_starvation_and_missing_snr(policy, ctx):
    hm = empty_frames(1)[0]
    starved = run_window(policy, ctx, None, hm, np.zeros((0, 7)))
    assert starved.penalties["point_snr"] == 1
    assert starved.penalties["point_count"] == 1
    assert "no ROI detections" in starved.warnings

    no_snr = run_window(policy, ctx, None, hm, occupied_points(snr=0.0), snr_available=False)
    assert no_snr.pointcloud_score is None and no_snr.reasons["pointcloud"] == "snr_missing"


def test_points_outside_roi_do_not_count_but_still_fill_the_cap(policy, ctx):
    near = np.array([[0.0, 0.1, 0.0, 0.0, 30.0, 30.0, 255.0]] * 10)  # 0.1 m, inside the leakage zone
    w = run_window(policy, ctx, None, empty_frames(1)[0], near)
    assert w.measurements["roi_points"] == 0
    assert w.measurements["all_points"] == 160
    assert w.penalties["point_cap"] == pytest.approx((160 - 128) / 160)


def test_declared_but_undecoded_points_are_not_an_empty_cloud(policy, ctx):
    frames = [q.measure_frame(empty_frames(1)[0], np.zeros((0, 7)), ctx, None, policy, i, "present",
                              declared_points=4) for i in range(policy.window_len)]
    w = q.score_window(frames, policy)
    assert w.penalties["point_count"] is None and w.reasons["pointcloud"] == "point_output_missing"


def test_reference_from_other_config_is_refused(policy, ctx, tmp_path):
    ref = q.fit_empty_reference(empty_frames(), ctx, policy)
    ref.save(tmp_path / "ref.npz")
    assert q.EmptyReference.load(tmp_path / "ref.npz", ctx).n_frames == 50
    other = q.build_context(CFG + ["clutterRemoval -1 1"], np.linspace(0, 6, R), D, policy.roi_range_m)
    with pytest.raises(ValueError, match="config"):
        q.EmptyReference.load(tmp_path / "ref.npz", other)


def test_config_hash_ignores_comments_and_whitespace():
    assert q.config_hash(["sensorStop", "% x", "frameCfg  0 2"]) == q.config_hash(["sensorStop\n", "frameCfg 0 2"])


def test_monitor_restarts_on_gap_and_invalid_frame(policy, ctx):
    mon = q.QualityMonitor(ctx, policy)
    hm, pts = empty_frames(1)[0], occupied_points()
    for n in range(1, 21):
        w = mon.push(n, hm, pts, "present")
    assert w.status == "ready"
    w = mon.push(23, hm, pts, "present")
    assert w.status == "warming_up" and w.frames == 1 and "gap" in mon.notice
    w = mon.push(24, hm[:10], pts, "present")
    assert w.frames == 0 and "wrong_heatmap_shape" in mon.notice


def test_monitor_collects_reference_then_scores_heatmap(policy, ctx):
    mon = q.QualityMonitor(ctx, policy)
    mon.start_reference()
    for n, hm in enumerate(empty_frames(policy.reference_frames), start=1):
        mon.push(n, hm, np.zeros((0, 7)), "empty")
    assert mon.reference is not None and mon.reference_progress is None
    assert mon.frames[-1].visibility is not None
