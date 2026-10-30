import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {BrowserDQN, BrowserRidge} from '../js/model.js';
import {gaPairScore, glennAloiPairScore, sortedMoves} from '../js/policy.js';
import {PendingActions} from '../js/pending.js';
import {randomSearchRollout} from '../js/vmerckle.js';

function seededRng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (1664525 * value + 1013904223) >>> 0;
    return value / 0x100000000;
  };
}

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

test('Ridge uses its feature_count schema instead of neural metadata', async () => {
  const source=await readFile(new URL('../js/app.js',import.meta.url),'utf8');
  assert.match(source,/function ridgeFeatures\(g,player,policy\)/);
  assert.match(source,/feature_count:count,feature_mode:mode/);
  assert.match(source,/ridgeFeatures\(g,player,ridge\)/);
  assert.doesNotMatch(source,/modelFeatures\(g,player,ridge\)/);
});

test('Ridge rejects feature vectors with the wrong length', () => {
  const model=new BrowserRidge({feature_count:2,mean:[0,0],scale:[1,1],weights:[1,1],bias:0});
  assert.throws(() => model.action([0]), /expected 2 values/);
});

test('quick test runs headless games and reports both win rates', async () => {
  const source=await readFile(new URL('../js/app.js',import.meta.url),'utf8');
  assert.match(source,/const testGame=new Game\(\)/);
  assert.match(source,/wins\[testGame\.winner\]\+\+/);
  assert.match(source,/Player A \$\{aRate\}%/);
  assert.match(source,/const testAgents=\[\$\('agent-a'\)\.value,\$\('agent-b'\)\.value\]/);
  assert.match(source,/if\(testAgents\.some\(name=>name==='human'\)\)/);
  assert.match(source,/Running test… \$\{i\+1\}\/\$\{count\} games — Player A: \$\{wins\[0\]\} wins, Player B: \$\{wins\[1\]\} wins/);
  assert.match(source,/Running test… 0\/\$\{count\} games — Player A: 0 wins, Player B: 0 wins/);
  assert.match(source,/setTimeout\(resolve,0\)/);
  assert.match(source,/const animate=true/);
  assert.match(source,/setTimeout\(resolve,animate\?60:0\)/);
});

test('game status names the player instead of addressing the user', async () => {
  const source=await readFile(new URL('../js/game.js',import.meta.url),'utf8');
  assert.match(source,/t\('playerTurn',\{player:1\}\)/);
  assert.match(source,/t\('playerTurn',\{player:this\.player\+1\}\)/);
  assert.doesNotMatch(source,/this\.message=t\('yourTurn'\)/);
});

test('canceled rollout searches stop before recording results', async () => {
  const source=await readFile(new URL('../js/app.js',import.meta.url),'utf8');
  assert.match(source,/const result=await searchRootActions[\s\S]*?if\(!result\|\|generation!==runGeneration\)return null;/);
  assert.match(source,/const stopResult=await fullGameRollout[\s\S]*?if\(generation!==runGeneration\)return false;/);
  assert.match(source,/const continueResult=await fullGameRollout[\s\S]*?if\(generation!==runGeneration\)return false;/);
});

test('vmerckle rollout is reproducible with an injected random stream', async () => {
  const state = {
    permanent: [Array(11).fill(0), Array(11).fill(0)],
    owner: Array(11).fill(-1),
    active: {7: 3}, player: 0, rolls: 1, phase: 'decision', winner: null,
  };
  const options = {yieldEvery: 0};
  const first = await randomSearchRollout(state, 1, 0, {...options, rng: seededRng(441)});
  const second = await randomSearchRollout(state, 1, 0, {...options, rng: seededRng(441)});
  assert.equal(first, second);
});

test('vmerckle rollout does not use the live state or random source after cancellation', async () => {
  const state = {
    permanent: [Array(11).fill(0), Array(11).fill(0)],
    owner: Array(11).fill(-1), active: {}, player: 0, rolls: 0,
    phase: 'decision', winner: null,
  };
  let randomCalls = 0;
  let cancelled = false;
  const result = await randomSearchRollout(state, 1, 0, {
    rng: () => { randomCalls++; return 0.25; },
    yieldEvery: 1,
    yieldControl: async () => { cancelled = true; },
    isCancelled: () => cancelled,
  });
  assert.equal(result, null);
  assert.ok(randomCalls > 0);
  assert.deepEqual(state.active, {});
});
