import test from 'node:test';
import assert from 'node:assert/strict';
import {legalMoves} from '../js/game.js';
import {randomSearchRollout, searchRootActions} from '../js/vmerckle.js';
import {referenceRandomSearchRollout} from '../tools/vmerckle_reference.mjs';

function seededRng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (1664525 * value + 1013904223) >>> 0;
    return value / 0x100000000;
  };
}

function makeActions() {
  const game = {
    permanent: [Array(11).fill(0), Array(11).fill(0)],
    owner: Array(11).fill(-1), active: {}, player: 0,
  };
  const moves = legalMoves([1, 2, 3, 4], game);
  return moves.flatMap(move => [0, 1].map(action => {
    const post = {...game, active: {...game.active}, phase: 'decision', rolls: 1, winner: null};
    for (const column of move) {
      post.active[column] = (post.active[column] ?? game.permanent[game.player][column - 2]) + 1;
    }
    return {move, action, post};
  }));
}

async function runRoot(name, rollout, rng) {
  const actions = makeActions();
  const total = 64;
  const wins = Array(actions.length).fill(0);
  const plays = Array(actions.length).fill(0);
  if (name === 'vmerckle_monte') {
    const quota = Math.floor(total / actions.length);
    for (let i = 0; i < actions.length; i++) {
      for (let j = 0; j < quota; j++) {
        wins[i] += await rollout(actions[i].post, actions[i].action, 0, rng);
      }
    }
  } else {
    for (let n = 0; n < total; n++) {
      let i = plays.findIndex(value => value === 0);
      if (i < 0) {
        i = plays.reduce((best, value, k) =>
          wins[k] / value + Math.sqrt(Math.log(n) / value) >
          wins[best] / plays[best] + Math.sqrt(Math.log(n) / plays[best]) ? k : best, 0);
      }
      wins[i] += await rollout(actions[i].post, actions[i].action, 0, rng);
      plays[i]++;
    }
  }
  const candidates = actions.map((action, i) => ({action, i}))
    .filter(candidate => plays[candidate.i] || name === 'vmerckle_monte');
  const best = candidates.reduce((x, y) =>
    wins[y.i] / Math.max(plays[y.i], 1) > wins[x.i] / Math.max(plays[x.i], 1) ? y : x,
    candidates[0]);
  return {index: best.i, wins, plays};
}

for (const name of ['vmerckle_monte', 'vmerckle_monte_uct']) {
  test(`${name} preserves root action selection`, async () => {
    const reference = await runRoot(name, (state, action, root, rng) =>
      referenceRandomSearchRollout(state, action, root, rng), seededRng(12345));
    const actions = makeActions();
    const optimized = await searchRootActions(actions, 0, name, {
      rng: seededRng(12345), yieldControl: async () => {},
      rollout: (state, action, root, options) => randomSearchRollout(
        state, action, root, {...options, yieldEvery: 0},
      ),
    });
    const optimizedComparable = {
      index: optimized.index, wins: optimized.wins, plays: optimized.plays,
    };
    assert.deepEqual(optimizedComparable, reference);
  });
}
