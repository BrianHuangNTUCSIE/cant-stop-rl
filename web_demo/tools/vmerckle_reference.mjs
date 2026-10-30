import {legalMoves, LENGTHS} from '../js/game.js';

// Reference implementation copied from the pre-optimization browser rollout.
// This file is test/benchmark-only and is intentionally not imported by app.js.
export async function referenceRandomSearchRollout(g, action, root, rng = Math.random) {
  const s = {
    permanent: g.permanent.map(row => row.slice()),
    owner: g.owner.slice(),
    active: {...g.active},
    player: g.player,
    rolls: g.rolls,
    phase: g.phase,
    winner: g.winner,
  };
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
  let turns = 0;
  while (turns < 2000 && s.winner === null) {
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
