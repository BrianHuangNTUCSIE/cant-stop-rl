// Small dependency-free MLP inference. Training remains in Python; only the
// online network is shipped to the browser.
export class BrowserDQN {
  constructor(payload) { this.payload=payload; this.layers=payload.layers; }
  static async load(url='models/dqn.json') {
    try { const r=await fetch(url); if(!r.ok) throw Error('missing model'); return new BrowserDQN(await r.json()); }
    catch (_) { return null; }
  }
  forward(input) {
    let x=input;
    for (let i=0;i<this.layers.length;i++) { const l=this.layers[i]; const y=new Array(l.bias.length); for(let r=0;r<y.length;r++){let v=l.bias[r];for(let c=0;c<x.length;c++)v+=l.weights[r][c]*x[c];y[r]=i===this.layers.length-1?v:(this.payload.activation==='tanh'?Math.tanh(v):Math.max(0,v))} x=y; }
    return x;
  }
  action(features) {
    const q=this.forward(features);
    // Joint PPO checkpoints reserve actions 0--5 for pairing and use 6/7 for
    // Continue/Stop.  The browser keeps GA pairing, so only the decision head
    // is consulted here.
    if(this.payload.action_size===8)return q[7]>q[6]?'stop':'continue';
    return q[1]>q[0]?'stop':'continue';
  }
}

export function heuristicAction(state) {
  const active=Object.entries(state.active);
  if (active.some(([column,position])=>position===state.lengths[column])) return 'stop';
  if (active.length<3) return 'continue';
  const points=active.reduce((sum,[column,position])=>sum+(position-state.permanent[0][column-2]+1)*state.weights[column],0);
  return points>=28?'stop':'continue';
}
