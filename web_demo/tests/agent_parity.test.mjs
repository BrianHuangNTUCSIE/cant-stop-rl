import test from 'node:test';
import assert from 'node:assert/strict';
import {BrowserDQN} from '../js/model.js';
import {gaPairScore, glennAloiPairScore, sortedMoves} from '../js/policy.js';

test('GA duplicate pair charges one new-marker penalty', () => {
  assert.equal(gaPairScore({}, [7, 7]), 0);
  assert.equal(gaPairScore({7: 2}, [7, 7]), 6);
});

test('Glenn & Aloi scores both advances of a duplicate pair', () => {
  assert.equal(glennAloiPairScore({}, [2, 2]), 8);
  assert.equal(glennAloiPairScore({2: 1}, [2, 2]), 14);
});

test('eight-action PPO selects the legal sorted pairing logits', () => {
  const model=new BrowserDQN({action_size:8, layers:[{weights:Array.from({length:8},(_,i)=>[i]), bias:Array(8).fill(0)}]});
  const moves=[[8,9],[2,12],[4,5]];
  const ordered=sortedMoves(moves);
  assert.deepEqual(model.pairAction([1], moves), ordered[2]);
});
