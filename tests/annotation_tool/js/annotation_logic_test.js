// Run with: node tests/annotation_tool/js/annotation_logic_test.js
const assert = require("assert");
const A = require("../../../src/annotation_tool/static/logic.js");

const flags = (n, ...runs) => A.fromIntervals(n, runs);
const arr = f => Array.from(f);

// intervals <-> flags
assert.deepStrictEqual(A.toIntervals(flags(10, [2, 3], [7, 9])), [[2, 3], [7, 9]]);
assert.deepStrictEqual(A.toIntervals(flags(5)), []);
assert.deepStrictEqual(A.toIntervals(flags(5, [0, 0], [4, 4])), [[0, 0], [4, 4]]);   // the first and the last frame
assert.deepStrictEqual(arr(A.fromIntervals(5, [[1, 2]])), [0, 1, 1, 0, 0]);
assert.deepStrictEqual(A.intervalAt(flags(10, [2, 5]), 4), [2, 5]);
assert.strictEqual(A.intervalAt(flags(10, [2, 5]), 6), null);

// edits never change the old array
const base = flags(10, [2, 3]);
const added = A.setRange(base, 3, 6, true);
assert.deepStrictEqual(A.toIntervals(added), [[2, 6]]);                              // joins the neighbour
assert.deepStrictEqual(A.toIntervals(base), [[2, 3]]);
assert.deepStrictEqual(A.toIntervals(A.setRange(base, 7, 5, true)), [[2, 3], [5, 7]]); // the order of the ends does not matter
assert.deepStrictEqual(A.toIntervals(A.setRange(base, 4, 5, true)), [[2, 5]]);          // touching intervals merge
assert.deepStrictEqual(A.toIntervals(A.toggle(base, 2)), [[3, 3]]);                  // toggling an end frame shortens it
assert.deepStrictEqual(A.toIntervals(A.toggle(flags(10, [2, 6]), 4)), [[2, 3], [5, 6]]); // toggling the middle splits it
assert.deepStrictEqual(A.toIntervals(A.toggle(base, 8)), [[2, 3], [8, 8]]);
assert.deepStrictEqual(A.toIntervals(A.clearIntervalAt(flags(10, [2, 3], [6, 8]), 7)), [[2, 3]]);
const untouched = flags(10, [2, 3]);
assert.strictEqual(A.clearIntervalAt(untouched, 9), untouched);                      // nothing to clear: the same array

// commands
let s = {flags: flags(10), reviewed: false, pending: null};
let r = A.finish(s, 4);
assert.strictEqual(r.changed, false);
assert.match(r.message, /press i first/);                                           // o before i does nothing
r = A.mark(s, 6); s = r.state;
assert.strictEqual(s.pending, 6); assert.strictEqual(r.changed, false);
r = A.finish(s, 3); s = r.state;                                                     // i at 6, o at 3: the ends are sorted
assert.deepStrictEqual(A.toIntervals(s.flags), [[3, 6]]);
assert.strictEqual(r.changed, true); assert.strictEqual(s.pending, null);
r = A.mark(s, 4); r = A.finish(r.state, 5);                                          // already lying: nothing changes
assert.strictEqual(r.changed, false);
r = A.removeInterval(s, 9);
assert.strictEqual(r.changed, false); assert.match(r.message, /no lying interval/);
r = A.removeInterval(s, 5);
assert.deepStrictEqual(A.toIntervals(r.state.flags), []); assert.strictEqual(r.changed, true);
r = A.toggleReviewed(s); assert.strictEqual(r.state.reviewed, true); assert.strictEqual(r.changed, true);
assert.strictEqual(A.toggleFrame(s, 0).changed, true);

// undo and redo
const h = new A.History(3);
const states = [0, 1, 2, 3, 4].map(i => ({flags: flags(5, [i, i]), reviewed: false}));
assert.strictEqual(h.undo(states[0]), null);                                         // nothing to undo yet
for (let i = 0; i < 4; i++) h.record(states[i]);                                     // states 0..3, the limit is 3
let current = states[4];
let back = h.undo(current);
assert.deepStrictEqual(A.toIntervals(back.flags), [[3, 3]]);
current = back;
back = h.undo(current); current = back;
back = h.undo(current); current = back;
assert.deepStrictEqual(A.toIntervals(current.flags), [[1, 1]]);                      // state 0 fell out of the history
assert.strictEqual(h.undo(current), null);
const forward = h.redo(current);
assert.deepStrictEqual(A.toIntervals(forward.flags), [[2, 2]]);
h.record(forward);                                                                   // a new change clears redo
assert.strictEqual(h.redo(forward), null);

console.log("annotation logic: all checks passed");
