import {randomSearchRollout} from '../js/vmerckle.js';
import {referenceRandomSearchRollout} from './vmerckle_reference.mjs';

const rollouts = Number.parseInt(process.argv[2] ?? '100', 10);
if (!Number.isInteger(rollouts) || rollouts < 1) {
  throw new Error('usage: node tools/benchmark_vmerckle.mjs [positive rollout count]');
}

function seededRng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (1664525 * value + 1013904223) >>> 0;
    return value / 0x100000000;
  };
}

const state = {
  permanent: [Array(11).fill(0), Array(11).fill(0)],
  owner: Array(11).fill(-1),
  active: {6: 4, 7: 5}, player: 0, rolls: 2,
  phase: 'decision', winner: null,
};

async function measure(rollout) {
  const start = performance.now();
  let wins = 0;
  for (let i = 0; i < rollouts; i++) {
    wins += await rollout(state, i % 2, 0, seededRng(441 + i));
  }
  return {elapsed: performance.now() - start, wins};
}

const reference = await measure((s, action, root, rng) =>
  referenceRandomSearchRollout(s, action, root, rng));
const optimized = await measure((s, action, root, rng) =>
  randomSearchRollout(s, action, root, {rng, yieldEvery: 0}));
const speedup = reference.elapsed / optimized.elapsed;
const result = {
  rollouts,
  reference: {
    elapsed_ms: Number(reference.elapsed.toFixed(3)),
    rollouts_per_second: Number((rollouts / (reference.elapsed / 1000)).toFixed(1)),
    wins: reference.wins,
  },
  optimized: {
    elapsed_ms: Number(optimized.elapsed.toFixed(3)),
    rollouts_per_second: Number((rollouts / (optimized.elapsed / 1000)).toFixed(1)),
    wins: optimized.wins,
  },
  equivalent_wins: reference.wins === optimized.wins,
  speedup: Number(speedup.toFixed(3)),
};
console.log(JSON.stringify(result));
