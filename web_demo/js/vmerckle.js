import {legalMoves, LENGTHS} from './game.js';

// The rollout deliberately keeps the browser implementation's existing
// object representation and operation order.  This module only makes the
// random source and scheduling injectable so performance work can be tested
// against a reproducible stream without changing the normal UI path.
export function cloneSearchState(g) {
  return {
    permanent: g.permanent.map(row => row.slice()),
    owner: g.owner.slice(),
    active: {...g.active},
    player: g.player,
    rolls: g.rolls,
    phase: g.phase,
    winner: g.winner,
  };
}

export async function randomSearchRollout(
  g, action, root, {
    rng = Math.random,
    yieldEvery = 32,
    yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)),
    isCancelled = () => false,
  } = {},
) {
  const s = cloneSearchState(g);
  if (action === 0) {
    for (const [c, p] of Object.entries(s.active)) {
      s.permanent[s.player][c - 2] = p;
      if (+p === LENGTHS[c]) s.owner[c - 2] = s.player;
    }
    if (s.owner.filter(x => x === s.player).length >= 3) {
      s.winner = s.player;
      s.phase = 'done';
      s.active = {};
      return s.winner === root ? 1 : 0;
    }
    s.active = {};
    s.rolls = 0;
    s.player = 1 - s.player;
    s.phase = 'roll';
  } else {
    s.phase = 'roll';
  }

  if (isCancelled()) return null;
  let turns = 0;
  while (turns < 2000 && s.winner === null) {
    // The caller batches rollouts and yields between batches.  Avoid an
    // additional timer at turn zero for every rollout, but keep periodic
    // yields for unusually long individual simulations.
    if (yieldEvery > 0 && turns > 0 && turns % yieldEvery === 0) {
      await yieldControl();
      if (isCancelled()) return null;
    }
    if (s.phase === 'roll') {
      const dice = Array.from({length: 4}, () => 1 + Math.floor(rng() * 6));
      const moves = legalMoves(dice, s);
      if (!moves.length) {
        s.active = {};
        s.rolls = 0;
        s.player = 1 - s.player;
        turns++;
        continue;
      }
      s.moves = moves;
      s.phase = 'pair';
    }
    if (s.phase === 'pair') {
      const m = s.moves[Math.floor(rng() * s.moves.length)];
      s.active = {...s.active};
      for (const c of m) {
        s.active[c] = (s.active[c] ?? s.permanent[s.player][c - 2]) + 1;
      }
      s.rolls++;
      s.phase = 'decision';
    }
    if (s.phase === 'decision') {
      if (rng() < 0.5) {
        for (const [c, p] of Object.entries(s.active)) {
          s.permanent[s.player][c - 2] = p;
          if (+p === LENGTHS[c]) s.owner[c - 2] = s.player;
        }
        if (s.owner.filter(x => x === s.player).length >= 3) s.winner = s.player;
        s.active = {};
        s.rolls = 0;
        s.player = 1 - s.player;
        s.phase = s.winner === null ? 'roll' : 'done';
        turns++;
      } else {
        s.phase = 'roll';
      }
    }
  }
  return s.winner === root ? 1 : 0;
}

export async function searchRootActions(actions, root, name, {
  total = 64,
  rollout = randomSearchRollout,
  rng = Math.random,
  yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)),
  isCancelled = () => false,
  onRollout = null,
} = {}) {
  const wins = Array(actions.length).fill(0);
  const plays = Array(actions.length).fill(0);
  let yields = 0;
  let rollouts = 0;
  const yieldBetween = async () => {
    yields++;
    await yieldControl();
  };
  const run = async i => {
    const result = await rollout(actions[i].post, actions[i].action, root, {
      rng,
      isCancelled,
      yieldEvery: 32,
    });
    if (isCancelled() || result === null) return false;
    wins[i] += result;
    // The flat Monte implementation historically left `plays` at zero;
    // preserve that observable bookkeeping distinction from UCT.
    if (name !== 'vmerckle_monte') plays[i]++;
    rollouts++;
    onRollout?.({index: i, result, rollouts});
    return true;
  };

  if (name === 'vmerckle_monte') {
    const quota = Math.floor(total / actions.length);
    for (let i = 0; i < actions.length; i++) {
      for (let j = 0; j < quota; j++) {
        if ((i * quota + j) % 4 === 0) await yieldBetween();
        if (!(await run(i))) return null;
      }
    }
  } else {
    for (let n = 0; n < total; n++) {
      let i = plays.findIndex(x => x === 0);
      if (i < 0) {
        i = plays.reduce((best, x, k) =>
          wins[k] / x + Math.sqrt(Math.log(n) / x) >
          wins[best] / plays[best] + Math.sqrt(Math.log(n) / plays[best]) ? k : best, 0);
      }
      if (n % 4 === 0) await yieldBetween();
      if (!(await run(i))) return null;
    }
  }

  const candidates = actions.map((a, i) => ({a, i}))
    .filter(x => plays[x.i] || name === 'vmerckle_monte');
  const best = candidates.reduce((x, y) =>
    wins[y.i] / Math.max(plays[y.i], 1) > wins[x.i] / Math.max(plays[x.i], 1) ? y : x,
    candidates[0]);
  return {action: best.a, index: best.i, wins, plays, yields, rollouts};
}
