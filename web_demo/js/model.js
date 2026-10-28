import {sortedMoves} from './policy.js';

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
    if(this.payload.action_size===8)return q[7]>q[6]?'stop':'continue';
    return q[1]>q[0]?'stop':'continue';
  }
  pairAction(features, moves) {
    if(this.payload.action_size!==8) return null;
    const logits=this.forward(features);
    const ordered=sortedMoves(moves);
    const index=logits.slice(0, Math.min(6, ordered.length)).reduce(
      (best, value, i) => value>logits[best] ? i : best, 0);
    return ordered[index] ?? null;
  }
}

// Browser export of the domain-feature StandardScaler + Ridge continuation
// margin model. Python stops when the predicted continuation margin is <= 0.
export class BrowserRidge {
  constructor(payload) { this.payload=payload; }
  static async load(url='models/ridge.json') {
    try { const r=await fetch(url); if(!r.ok) throw Error('missing model'); return new BrowserRidge(await r.json()); }
    catch (_) { return null; }
  }
  action(features) {
    const {mean,scale,weights,bias}=this.payload;
    if(!Array.isArray(features)||features.length!==this.payload.feature_count||
       mean.length!==features.length||scale.length!==features.length||weights.length!==features.length)
      throw new Error(`Invalid Ridge feature vector: expected ${this.payload.feature_count} values`);
    let value=bias;
    for(let i=0;i<features.length;i++) value+=weights[i]*((features[i]-mean[i])/(scale[i]||1));
    return value<=0?'stop':'continue';
  }
}

export function heuristicAction(state) {
  const active=Object.entries(state.active);
  if (active.some(([column,position])=>position===state.lengths[column])) return 'stop';
  if (active.length<3) return 'continue';
  const points=active.reduce((sum,[column,position])=>sum+(position-state.permanent[0][column-2]+1)*state.weights[column],0);
  return points>=28?'stop':'continue';
}
