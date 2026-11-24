"use strict";
// The editing logic of the annotation page, without any screen code, so that it can be tested with node.
// A recording's marks are `flags`: a Uint8Array with one entry per frame in capture order, 1 = lying.
// What is saved is the list of lying intervals [first, last] (both frames included) made from them.

const Annotation = (() => {
  // [[first, last], ...] of every run of lying frames
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

  function fromIntervals(length, intervals) {
    const flags = new Uint8Array(length);
    for (const [first, last] of intervals) flags.fill(1, first, last + 1);
    return flags;
  }

  // the interval that contains frame `i`, or null
  function intervalAt(flags, i) {
    if (!flags[i]) return null;
    let a = i, b = i;
    while (a > 0 && flags[a - 1]) a--;
    while (b + 1 < flags.length && flags[b + 1]) b++;
    return [a, b];
  }

  // every edit returns a new array, so a snapshot of the old one stays valid for undo
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

  // undo and redo over snapshots {flags, reviewed}
  class History {
    constructor(limit = 50) { this.limit = limit; this.past = []; this.future = []; }
    // call with the state *before* a change
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

  // The editing commands. `state` is {flags, reviewed, pending}; each command returns
  // {state, changed, message}. `changed` is true when the saved marks change.
  function mark(state, pos) {  // key i: the first lying frame
    return {state: {...state, pending: pos}, changed: false, message: "start at " + pos + ", press o at the last lying frame"};
  }
  function finish(state, pos) {  // key o: the last lying frame
    if (state.pending === null) return {state, changed: false, message: "press i first, at the first lying frame"};
    const flags = setRange(state.flags, state.pending, pos, true);
    return {state: {...state, flags, pending: null}, changed: !same(flags, state.flags), message: ""};
  }
  function toggleFrame(state, pos) {  // key x
    return {state: {...state, flags: toggle(state.flags, pos)}, changed: true, message: ""};
  }
  function removeInterval(state, pos) {  // key Delete
    const flags = clearIntervalAt(state.flags, pos);
    return {state: {...state, flags}, changed: flags !== state.flags, message: flags === state.flags ? "no lying interval here" : ""};
  }
  function toggleReviewed(state) {  // key r
    return {state: {...state, reviewed: !state.reviewed}, changed: true, message: ""};
  }

  return {toIntervals, fromIntervals, intervalAt, setRange, toggle, clearIntervalAt, History,
          mark, finish, toggleFrame, removeInterval, toggleReviewed};
})();

if (typeof module !== "undefined") module.exports = Annotation;
