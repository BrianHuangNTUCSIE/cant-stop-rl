import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {BrowserDQN} from '../js/model.js';
import {gaPairScore, glennAloiPairScore, sortedMoves} from '../js/policy.js';
import {PendingActions} from '../js/pending.js';

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

test('vmerckle pending action is consumed from the post-move state', () => {
  const pending=new PendingActions();
  const before={active:{},permanent:[[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]};
  const after={active:{7:1},permanent:before.permanent};
  pending.set(after,0,true);
  assert.equal(pending.consume(before,0),false);
  assert.equal(pending.consume(after,0),true);
  assert.equal(pending.consume(after,0),false);
});

test('domain feature construction passes the acting player to Rule 28', async () => {
  const source=await readFile(new URL('../js/app.js',import.meta.url),'utf8');
  assert.match(source,/rule28Points\(\{\.\.\.g,active:g\.active\},player\)/);
});

test('canceled rollout searches stop before recording results', async () => {
  const source=await readFile(new URL('../js/app.js',import.meta.url),'utf8');
  assert.match(source,/const result=await randomSearchRollout[\s\S]*?if\(generation!==runGeneration\)return null;/);
  assert.match(source,/const stopResult=await fullGameRollout[\s\S]*?if\(generation!==runGeneration\)return false;/);
  assert.match(source,/const continueResult=await fullGameRollout[\s\S]*?if\(generation!==runGeneration\)return false;/);
});
