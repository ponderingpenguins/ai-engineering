"use strict";
// Annotation editing rules shared by the browser UI and Node.js tests.
// Flags contain one byte per frame in capture order: 1 means lying, 0 means not lying.
// CSV saves use inclusive [first, last] intervals rather than individual frame flags.

const Annotation = (() => {
  /** Convert frame flags into inclusive [first, last] lying intervals. */
  function toIntervals(flags) {
    const out = [];
    for (let i = 0; i < flags.length; i++) {
      if (!flags[i]) continue;
      let j = i;
      while (j + 1 < flags.length && flags[j + 1]) j++;
      out.push([i, j]);
      i = j;
    }
    return out;
  }

  /** Build frame flags from inclusive intervals for a recording of the given length. */
  function fromIntervals(length, intervals) {
    const flags = new Uint8Array(length);
    for (const [first, last] of intervals) flags.fill(1, first, last + 1);
    return flags;
  }

  /** Return the inclusive lying interval containing frame i, or null. */
  function intervalAt(flags, i) {
    if (!flags[i]) return null;
    let a = i, b = i;
    while (a > 0 && flags[a - 1]) a--;
    while (b + 1 < flags.length && flags[b + 1]) b++;
    return [a, b];
  }

  /** Return a copy with both endpoints included, preserving the original for undo. */
  function setRange(flags, a, b, value) {
    const out = flags.slice();
    out.fill(value ? 1 : 0, Math.min(a, b), Math.max(a, b) + 1);
    return out;
  }
  const toggle = (flags, i) => setRange(flags, i, i, !flags[i]);
  function clearIntervalAt(flags, i) {
    const found = intervalAt(flags, i);
    return found ? setRange(flags, found[0], found[1], false) : flags;
  }

  const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

  // Undo and redo store snapshots shaped as {flags, reviewed}.
  class History {
    constructor(limit = 50) { this.limit = limit; this.past = []; this.future = []; }
    // Record the state before applying a change; a new edit clears redo history.
    record(state) {
      this.past.push(state);
      if (this.past.length > this.limit) this.past.shift();
      this.future = [];
    }
    undo(current) {
      if (!this.past.length) return null;
      this.future.push(current);
      return this.past.pop();
    }
    redo(current) {
      if (!this.future.length) return null;
      this.past.push(current);
      return this.future.pop();
    }
  }

  // Commands accept {flags, reviewed, pending} and return {state, changed, message}.
  // changed indicates whether saved marks changed; pending is an unsaved interval start.
  function mark(state, pos) {  // Key i: set the first lying frame.
    return {state: {...state, pending: pos}, changed: false, message: "start at " + pos + ", press o at the last lying frame"};
  }
  function finish(state, pos) {  // Key o: include the final lying frame and finish the interval.
    if (state.pending === null) return {state, changed: false, message: "press i first, at the first lying frame"};
    const flags = setRange(state.flags, state.pending, pos, true);
    return {state: {...state, flags, pending: null}, changed: !same(flags, state.flags), message: ""};
  }
  function toggleFrame(state, pos) {  // Key x: toggle the current frame's lying flag.
    return {state: {...state, flags: toggle(state.flags, pos)}, changed: true, message: ""};
  }
  function removeInterval(state, pos) {  // Key Delete: remove the interval containing this frame.
    const flags = clearIntervalAt(state.flags, pos);
    return {state: {...state, flags}, changed: flags !== state.flags, message: flags === state.flags ? "no lying interval here" : ""};
  }
  function toggleReviewed(state) {  // Key r: toggle whether the whole recording is reviewed.
    return {state: {...state, reviewed: !state.reviewed}, changed: true, message: ""};
  }

  return {toIntervals, fromIntervals, intervalAt, setRange, toggle, clearIntervalAt, History,
          mark, finish, toggleFrame, removeInterval, toggleReviewed};
})();

if (typeof module !== "undefined") module.exports = Annotation;
