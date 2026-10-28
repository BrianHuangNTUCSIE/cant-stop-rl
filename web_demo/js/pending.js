export function stateKey(game, player) {
  return `${player}:${Object.entries(game.active).sort((a, b) => a[0] - b[0]).map(([column, position]) => `${column}=${position}`).join(',')}`;
}

export class PendingActions {
  constructor() { this.actions = new Map(); }
  set(game, player, stop) { this.actions.set(stateKey(game, player), stop); }
  consume(game, player) {
    const key = stateKey(game, player);
    const action = this.actions.get(key);
    this.actions.delete(key);
    return action ?? false;
  }
  clear() { this.actions.clear(); }
}
