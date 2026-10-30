import test from 'node:test';
import assert from 'node:assert/strict';
import {randomSearchRollout} from '../js/vmerckle.js';
import {referenceRandomSearchRollout} from '../tools/vmerckle_reference.mjs';

function seededRng(seed, trace = null) {
  let value = seed >>> 0;
  return () => {
    value = (1664525 * value + 1013904223) >>> 0;
    const result = value / 0x100000000;
    trace?.push(result);
    return result;
  };
}

function state(seed) {
  return {
    permanent: [
      Array.from({length: 11}, (_, i) => (seed + i) % 3),
      Array.from({length: 11}, (_, i) => (seed + i * 2) % 3),
    ],
    owner: Array(11).fill(-1),
    active: {4: 2, 7: 3},
    player: seed % 2,
    rolls: 1,
    phase: 'decision',
    winner: null,
  };
}

for (const action of [0, 1]) {
  test(`optimized rollout matches reference for action ${action}`, async () => {
    const referenceTrace = [];
    const optimizedTrace = [];
    const reference = await referenceRandomSearchRollout(
      state(441), action, 0, seededRng(9001, referenceTrace),
    );
    const optimized = await randomSearchRollout(
      state(441), action, 0, {
        rng: seededRng(9001, optimizedTrace),
        yieldEvery: 0,
      },
    );
    assert.equal(optimized, reference);
    assert.deepEqual(optimizedTrace, referenceTrace);
  });
}

test('optimized rollout matches reference across representative states', async () => {
  for (const seed of [1, 7, 23, 101, 2026]) {
    for (const action of [0, 1]) {
      const referenceTrace = [];
      const optimizedTrace = [];
      const reference = await referenceRandomSearchRollout(
        state(seed), action, seed % 2, seededRng(seed + 500, referenceTrace),
      );
      const optimized = await randomSearchRollout(
        state(seed), action, seed % 2, {
          rng: seededRng(seed + 500, optimizedTrace),
          yieldEvery: 0,
        },
      );
      assert.equal(optimized, reference, `seed=${seed}, action=${action}`);
      assert.deepEqual(optimizedTrace, referenceTrace, `seed=${seed}, action=${action}`);
    }
  }
});
