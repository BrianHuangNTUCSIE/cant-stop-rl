import {Game, chooseGA, legalMoves, COLUMNS, LENGTHS, bustProbability} from './game.js';
import {BrowserDQN, heuristicAction} from './model.js';
import {gaPairScore, glennAloiPairScore, sortedMoves} from './policy.js';
import {PendingActions} from './pending.js';
import {detectLocale, setLocale, t} from './i18n.js';

setLocale(detectLocale());

const AGENTS = [
  ['human',''], ['random',''], ['rule28',''],
  ['rule28-smart','Smart Rule of 28'], ['glenn-aloi','Glenn & Aloi'],
  ['glenn2009_constant','Glenn 2009 constant'], ['glenn2010_linear','Glenn 2010 linear'],
  ['couto','Couto'], ['couto-ga-stop','Couto + GA pairing'],
  ['rule28-ga-pure','Rule of 28 + GA pairing'], ['mc','Monte Carlo'],
  ['mc-full-game','Full-game Monte Carlo'], ['mc-strong','MC strong'],
  ['vmerckle_monte','vmerckle Monte Carlo'],
  ['vmerckle_monte_uct','vmerckle root UCB1'], ['dqn','Trained DDQN'], ['ppo','Trained PPO']
];
const $=id=>document.getElementById(id); const game=new Game();
let model=null, ppo=null, dqnLoading=true, ppoLoading=true, agents=['human','rule28-smart'];
const label=name=>t(`agent.${name}`);
function localizeUI(){
  $('subtitle').textContent=t('subtitle'); $('new-game').textContent=t('newGame');
  $('model-label').textContent=t('loadingModel'); $('player-a-label').childNodes[0].textContent=t('playerA');
  $('player-b-label').childNodes[0].textContent=t('playerB'); $('apply-agents').textContent=t('applyAgents');
  $('board-label').textContent=t('board'); $('roll').textContent=t('roll'); $('stop').textContent=t('stop');
  $('continue').textContent=t('continue'); $('model-info').textContent=t('modelInfo');
}
function setupSelectors(){for(const id of ['agent-a','agent-b']){$(id).innerHTML=AGENTS.map(([value])=>`<option value="${value}">${label(value)}</option>`).join('');$(id).value=agents[id==='agent-a'?0:1];}updateModelOptions();}
function updateModelOptions(){
  for(const select of [$('agent-a'),$('agent-b')]){
    const option=select.querySelector('option[value="ppo"]');
    if(option){option.disabled=!ppo;option.textContent=ppo?label('ppo'):t('ppoUnavailable');}
    if(!ppo&&select.value==='ppo'){select.value='rule28-smart';}
  }
}
function modelState(value, loading){return value?t('loaded'):loading?t('loading'):t('unavailable');}
function modelStatus(){return t('modelsStatus',{dqn:!model&&!dqnLoading?t('dqnMissing'):modelState(model,dqnLoading),ppo:modelState(ppo,ppoLoading)});}
function isAI(player){return agents[player]!=='human';}
function render(){
  $('turn-label').textContent=game.phase==='done'?t('gameOver'):t('turn',{player:label(agents[game.player])});
  $('message').textContent=game.message; $('model-label').textContent=modelStatus(); $('roll').disabled=game.phase!=='roll'||game.winner!==null||isAI(game.player);
  $('stop').disabled=game.phase!=='decision'||isAI(game.player); $('continue').disabled=game.phase!=='decision'||isAI(game.player);
  $('neutral-label').textContent=t('neutral',{count:Object.keys(game.active).length});
  $('dice').innerHTML=game.dice.map(n=>`<span class="die">${n}</span>`).join('');
  $('moves').innerHTML=game.phase==='pair'&&!isAI(game.player)?game.moves.map((m,i)=>`<button class="move" data-move="${i}">${m.join(' + ')}</button>`).join(''):'';
  document.querySelectorAll('[data-move]').forEach(b=>b.onclick=()=>{game.applyMove(game.moves[+b.dataset.move]);render()});
  $('board').innerHTML=COLUMNS.map((c,i)=>{
    const height=LENGTHS[c], a=game.permanent[0][i], b=game.permanent[1][i];
    let html='';
    for(let p=1;p<=height;p++){
      const aBank=a>=p, bBank=b>=p;
      const active=game.active[c]||0;
    const marker=active===p ? `<span class="marker marker-${game.player===0?'a':'b'}" title="${t('marker',{player:game.player+1})}" aria-label="${t('marker',{player:game.player+1})}"></span>` : '';
      const fill=aBank&&bBank?'filled-both':aBank?'filled-a':bBank?'filled-b':'';
      html+=`<div class="cell ${fill} ${game.owner[i]!==-1?'closed':''}">${marker}</div>`;
    }
    return `<div class="column"><h2>${c}</h2><div class="track">${html}</div><small>A ${a}/${height} · B ${b}/${height}</small></div>`;
  }).join('');
   $('debug').textContent=JSON.stringify({players:agents.map(label),phase:game.phase,neutralMarkers:Object.keys(game.active).length,active:game.active,models:{dqn:!!model,ppo:!!ppo},modelFiles:{dqn:'models/dqn.json',ppo:'models/ppo.json'}},null,2);
}
const GA={2:7,3:0,4:2,5:0,6:4,7:3,8:4,9:0,10:2,11:0,12:7};
const CLASSIC={2:1,3:2,4:3,5:4,6:5,7:6,8:5,9:4,10:3,11:2,12:1};
const PUBLISHED={2:7,3:7,4:3,5:2,6:2,7:1,8:2,9:2,10:3,11:7,12:7};
const MARK_VALUE={2:12,3:10,4:8,5:6,6:4,7:2,8:4,9:6,10:8,11:10,12:12};
const STEP_VALUE={2:6,3:5,4:4,5:3,6:2,7:1,8:2,9:3,10:4,11:5,12:6};
const GLENN10_PROGRESS={2:15,3:13,4:8,5:8,6:4,7:3,8:4,9:8,10:8,11:13,12:15};
const GLENN10_SLOPE={2:72,3:48,4:56,5:44,6:28,7:32,8:28,9:44,10:56,11:48,12:72};
const GLENN10_INTERCEPT={2:15,3:0,4:3,5:0,6:7,7:5,8:7,9:0,10:3,11:0,12:15};
function pairScore(g,m,kind='ga'){
  const w=kind==='couto'?{2:0,3:1,4:2,5:3,6:4,7:5,8:4,9:3,10:2,11:1,12:0}:kind==='classic'?CLASSIC:GA;
  // A duplicated pair advances one neutral marker twice, but opening that
  // marker incurs the penalty only once.  This is the Python pair_score/
  // _best_pair_moves rule and matters for new doubles.
  const newMarkers=new Set(m.filter(c=>g.active[c]===undefined));
  return m.reduce((s,c)=>s+w[c],0)-6*newMarkers.size;
}
function bestMoves(moves,scores){const best=Math.max(...scores);return moves.filter((_,i)=>scores[i]===best)}
async function choosePair(g,moves,name){
  const generation=runGeneration;
  if(name==='ppo'){
    const learned=chooseLearnedPair(g,moves);
    if(learned)return learned;
  }
  if(name==='random')return moves[Math.floor(Math.random()*moves.length)];
  if(name==='vmerckle_monte'||name==='vmerckle_monte_uct'){
    const actions=[];
    for(const move of moves){
      const post={...g,active:{...g.active},phase:'decision'};
      for(const c of move)post.active[c]=(post.active[c]??g.permanent[g.player][c-2])+1;
      for(const action of [0,1])actions.push({move,action,post});
    }
    const total=64, wins=Array(actions.length).fill(0), plays=Array(actions.length).fill(0);
    if(name==='vmerckle_monte'){
      const quota=Math.floor(total/actions.length);
      for(let i=0;i<actions.length;i++)for(let j=0;j<quota;j++){
        if((i*quota+j)%4===0)await new Promise(resolve=>setTimeout(resolve,0));
        if(generation!==runGeneration)return null;
        const result=await randomSearchRollout(actions[i].post,actions[i].action,g.player);
        if(generation!==runGeneration)return null;
        wins[i]+=result;
      }
    }else{
      for(let n=0;n<total;n++){
        let i=plays.findIndex(x=>x===0);
        if(i<0)i=plays.reduce((best,x,k)=>wins[k]/x+Math.sqrt(Math.log(n)/x)>wins[best]/plays[best]+Math.sqrt(Math.log(n)/plays[best])?k:best,0);
        if(n%4===0)await new Promise(resolve=>setTimeout(resolve,0));
        if(generation!==runGeneration)return null;
        const result=await randomSearchRollout(actions[i].post,actions[i].action,g.player);
        if(generation!==runGeneration)return null;
        wins[i]+=result;plays[i]++;
      }
    }
    const candidates=actions.map((a,i)=>({a,i})).filter(x=>plays[x.i]||name==='vmerckle_monte');
    const best=candidates.reduce((x,y)=>wins[y.i]/Math.max(plays[y.i],1)>wins[x.i]/Math.max(plays[x.i],1)?y:x,candidates[0]);
    pendingVmerckle.set(best.a.post,g.player,best.a.action===0);
    return best.a.move;
  }
  if(name==='glenn-aloi')return moves.reduce((best,m)=>{
    const score=glennAloiPairScore(g.active,m);
    const old=best?best.score:-Infinity; return score>old?{move:m,score}:best;
  },null).move;
  if(name==='glenn2009_constant')return moves.reduce((best,m)=>{
    const score=gaPairScore(g.active,m);
    return !best||score>best.score?{move:m,score}:best;
  },null).move;
  if(name==='glenn2010_linear')return moves.reduce((best,m)=>{
    let score=0; for(const c of new Set(m)){const start=g.active[c]??g.permanent[g.player][c-2];const end=start+m.filter(x=>x===c).length;
      for(let p=start+1;p<=end;p++)score+=Math.floor(GLENN10_SLOPE[c]*p/LENGTHS[c]+GLENN10_INTERCEPT[c]);
      if(g.active[c]===undefined)score-=12;
    } return !best||score>best.score?{move:m,score}:best;
  },null).move;
  const kind=name==='couto'?'couto':name==='rule28'?'classic':'ga';
  const scores=moves.map(m=>{
    if(name==='couto')return m.reduce((s,c)=>s+(g.owner[c-2]===g.player?LENGTHS[c]+1:g.owner[c-2]===1-g.player?0:(g.active[c]??g.permanent[g.player][c-2]))+GA[c],0);
    return pairScore(g,m,kind);
  });
  const choices=bestMoves(moves,scores);
  // Rule28 and the GA-fixed adaptation use the Python agent's randomized tie;
  // paper agents use first-max tie breaking.
  if(['couto','glenn-aloi','glenn2009_constant','glenn2010_linear'].includes(name))return choices[0];
  return choices[Math.floor(Math.random()*choices.length)];
}
function paperDifficulty(cols,odds,evens,highs,lows){if(!cols.length)return 0;let s=0;if(cols.every(c=>c%2))s+=odds;if(cols.every(c=>c%2===0))s+=evens;if(cols.every(c=>c>=7))s+=highs;if(cols.every(c=>c<=7))s+=lows;return s}
function winsAfterStop(g,player){return g.permanent[player].filter((_,i)=>g.owner[i]===player).length+Object.entries(g.active).filter(([c,p])=>+p===LENGTHS[c]).length>=3}
 function rule28Points(g,player){const me=g.permanent[player], active=Object.entries(g.active);let s=0;for(const [c,p] of active)s+=MARK_VALUE[c]+(p-me[c-2]-1)*STEP_VALUE[c];const cols=active.map(([c])=>+c);if(cols.length===3){if(cols.every(c=>c%2))s+=2;if(cols.every(c=>c%2===0))s-=2;if(cols.every(c=>c<8))s+=4;if(cols.every(c=>c>6))s+=4;}return s}
function rule28ShouldStop(g,player){
  if(Object.entries(g.active).some(([c,p])=>+p===LENGTHS[c]))return true;
  return Object.keys(g.active).length>=3&&rule28Points(g,player)>=28;
}
function strongBankValue(g,player,active){
  const me=g.permanent[player], opponent=1-player;
  let value=0, claimed=0;
  for(const [column,position] of Object.entries(active)){
    value+=(position-me[column-2])/LENGTHS[column];
    if(+position===LENGTHS[column])claimed++;
  }
  value+=claimed*1.5;
  if(g.owner.filter(x=>x===player).length+claimed>=3)value+=20;
  let ownProgress=0, opponentProgress=0;
  for(const [column,position] of Object.entries(active))
    ownProgress+=(position-me[column-2])/LENGTHS[column];
  for(let i=0;i<COLUMNS.length;i++)opponentProgress+=g.permanent[opponent][i]/LENGTHS[COLUMNS[i]];
  opponentProgress/=11;
  value+=1.5*(g.owner.filter(x=>x===player).length+claimed-g.owner.filter(x=>x===opponent).length);
  value+=0.5*(ownProgress-opponentProgress);
  if(g.owner.filter(x=>x===opponent).length>=2 && g.owner.filter(x=>x===player).length+claimed<2)value-=1;
  return value;
}
function mcStrongStop(g,player){
  const rollouts=64, futureRollLimit=40;
  const rollout=()=>{
    const active={...g.active};
    const sim={active,owner:g.owner,permanent:g.permanent,player};
    for(let step=0;step<futureRollLimit;step++){
      const dice=Array.from({length:4},()=>1+Math.floor(Math.random()*6));
      sim.active=active;
      const moves=legalMoves(dice,sim);
      if(!moves.length)return 0;
      const selected=chooseGA(sim,moves);
      for(const column of selected)active[column]=(active[column]??g.permanent[player][column-2])+1;
      sim.active=active;
      if(rule28ShouldStop(sim,player))return strongBankValue(g,player,active);
    }
    return strongBankValue(g,player,active);
  };
  const bankNow=strongBankValue(g,player,g.active);
  let continueValue=0;
  for(let i=0;i<rollouts;i++)continueValue+=rollout();
  return bankNow>=continueValue/rollouts;
}
// Python's ``mc`` uses the same one-turn search as mc-strong, but its value
// function is deliberately local (mc-strong adds race awareness).  Keeping
// this separate avoids silently making the two selectable agents identical.
async function mcStop(g,player,name){
  if(name==='mc-strong')return mcStrongStop(g,player);
  if(name==='mc-full-game'){
    // Python's FullGameMonteCarloAgent uses GA pairing and Smart Rule 28 in
    // the remainder of each rollout.  vmerckle's separate rollout below is
    // intentionally random, so do not reuse it here.
    const generation=runGeneration;
    let stop=0,cont=0;for(let i=0;i<64;i++){
      if(i%2===0)await new Promise(resolve=>setTimeout(resolve,0));
      if(generation!==runGeneration)return false;
      const stopResult=await fullGameRollout(g,1,player);
      if(generation!==runGeneration)return false;
      stop+=stopResult;
      const continueResult=await fullGameRollout(g,0,player);
      if(generation!==runGeneration)return false;
      cont+=continueResult;
    }
    return stop>=cont;
  }
  const rollouts=64, limit=40, me=g.permanent[player];
  const bankValue=active=>{
    let value=0, claimed=0;
    for(const [c,p] of Object.entries(active)){
      value+=(p-me[c-2])/LENGTHS[c]; if(+p===LENGTHS[c])claimed++;
    }
    value+=claimed*1.5;
    if(g.owner.filter(x=>x===player).length+claimed>=3)value+=20;
    return value;
  };
  const rollout=()=>{
    let active={...g.active};
    for(let step=0;step<limit;step++){
      const dice=Array.from({length:4},()=>1+Math.floor(Math.random()*6));
      const sim={...g,active,permanent:g.permanent,owner:g.owner,player};
      const moves=legalMoves(dice,sim); if(!moves.length)return 0;
      const selected=chooseGA(sim,moves);
      active={...active}; for(const c of selected)active[c]=(active[c]??me[c-2])+1;
      const points=rule28Points({...g,active},player);
      if(Object.entries(active).some(([c,p])=>p===LENGTHS[c]) ||
         (Object.keys(active).length>=3&&points>=28))return bankValue(active);
    }
    return bankValue(active);
  };
  let future=0;for(let i=0;i<rollouts;i++){if(i%4===0)await new Promise(resolve=>setTimeout(resolve,0));future+=rollout();}
  return bankValue(g.active)>=future/rollouts;
}

// FullGameMonteCarloAgent's private simulator.  `action` follows Python's
// convention: 1 means bank now and 0 means continue.  This is deliberately
// separate from randomSearchRollout, whose random pairing/action policy is
// the vmerckle AIrand adaptation.
async function fullGameRollout(g,action,root){
  const generation=runGeneration;
  const s=cloneForSearch(g);
  if(action===1){
    for(const [c,p] of Object.entries(s.active)){
      s.permanent[s.player][c-2]=p;
      if(+p===LENGTHS[c])s.owner[c-2]=s.player;
    }
    if(s.owner.filter(x=>x===s.player).length>=3){s.winner=s.player;s.phase='done';s.active={};return s.winner===root?1:0;}
    s.active={};s.rolls=0;s.player=1-s.player;s.phase='roll';
  }else s.phase='roll';
  let turns=0;
  while(s.winner===null&&turns<2000){
    if(turns%8===0){await new Promise(resolve=>setTimeout(resolve,0));if(generation!==runGeneration)return null;}
    if(s.phase==='roll'){
      const dice=Array.from({length:4},()=>1+Math.floor(Math.random()*6));
      const moves=legalMoves(dice,s);
      if(!moves.length){s.active={};s.rolls=0;s.player=1-s.player;turns++;continue;}
      s.moves=moves;s.phase='pair';
    }
    if(s.phase==='pair'){
      const m=chooseGA(s,s.moves);
      s.active={...s.active};
      for(const c of m)s.active[c]=(s.active[c]??s.permanent[s.player][c-2])+1;
      s.rolls++;s.phase='decision';
    }
    if(s.phase==='decision'){
      if(rule28ShouldStop(s,s.player)){
        for(const [c,p] of Object.entries(s.active)){
          s.permanent[s.player][c-2]=p;
          if(+p===LENGTHS[c])s.owner[c-2]=s.player;
        }
        if(s.owner.filter(x=>x===s.player).length>=3){s.winner=s.player;s.phase='done';}
        s.active={};s.rolls=0;
        if(s.winner===null){s.player=1-s.player;s.phase='roll';}
        turns++;
      }else s.phase='roll';
    }
  }
  return s.winner===root?1:0;
}

// The Python vmerckle agents search the root action space (move x
// stop/continue), unlike the old browser fallback which used a race score.
// This compact implementation preserves that action-space distinction and
// the flat-vs-UCB1 selection; rollouts use the same random full-game policy.
const pendingVmerckle=new PendingActions();
function cloneForSearch(g){return {permanent:g.permanent.map(x=>x.slice()),owner:g.owner.slice(),active:{...g.active},player:g.player,rolls:g.rolls,phase:g.phase,winner:g.winner};}
async function randomSearchRollout(g,action,root){
  const generation=runGeneration;
  const s=cloneForSearch(g); if(action===0){
    for(const [c,p] of Object.entries(s.active)){s.permanent[s.player][c-2]=p;if(+p===LENGTHS[c])s.owner[c-2]=s.player;}
    if(s.owner.filter(x=>x===s.player).length>=3){s.winner=s.player;s.phase='done';s.active={};return s.winner===root?1:0;}
    s.active={};s.rolls=0;s.player=1-s.player;s.phase='roll';
  } else s.phase='roll';
  let turns=0;
  while(turns<2000&&s.winner===null){
    if(turns%8===0){await new Promise(resolve=>setTimeout(resolve,0));if(generation!==runGeneration)return null;}
    if(s.phase==='roll'){
      const dice=Array.from({length:4},()=>1+Math.floor(Math.random()*6));
      const moves=legalMoves(dice,s);if(!moves.length){s.active={};s.rolls=0;s.player=1-s.player;turns++;continue;}
      s.moves=moves;s.phase='pair';
    }
    if(s.phase==='pair'){const m=s.moves[Math.floor(Math.random()*s.moves.length)];s.active={...s.active};for(const c of m)s.active[c]=(s.active[c]??s.permanent[s.player][c-2])+1;s.rolls++;s.phase='decision';}
    if(s.phase==='decision'){
     if(Math.random()<.5){for(const [c,p] of Object.entries(s.active)){s.permanent[s.player][c-2]=p;if(+p===LENGTHS[c])s.owner[c-2]=s.player;}if(s.owner.filter(x=>x===s.player).length>=3)s.winner=s.player;s.active={};s.rolls=0;s.player=1-s.player;s.phase=s.winner===null?'roll':'done';turns++;}
      else s.phase='roll';
    }
  }
  return s.winner===root?1:0;
}
function vmerckleStop(g,player,name){
  // Search is scheduled at the pair stage; the decision stage consumes the
  // selected root action, matching Python's pending-action protocol.
  return pendingVmerckle.consume(g,player);
}
async function stopDecision(g,name,player){
  const active=Object.entries(g.active), me=g.permanent[player], opp=g.permanent[1-player];
  if(name==='random')return Math.random()<.3;
  if(name==='rule28')return active.length>=3&&rule28Points(g,player)>=28;
  // The plain Python Rule28Agent (smart=False) does not force a stop merely
  // because a neutral reached the top; only the smart variant does so.
  if(name==='rule28-smart')return active.some(([c,p])=>+p===LENGTHS[c])||(active.length>=3&&rule28Points(g,player)>=28);
  if(name==='rule28-ga-pure')return active.length>=3&&rule28Points(g,player)>=28;
  if(winsAfterStop(g,player))return true;
  if(name==='glenn-aloi'){
    if(active.length<3&&g.owner.some(x=>x===-1))return false;
    return active.reduce((s,[c,p])=>s+(+p-me[c-2]+1)*PUBLISHED[c],0)+paperDifficulty(active.map(([c])=>+c),7,1,6,5)>=29;
  }
  if(name==='glenn2009_constant')return active.reduce((s,[c,p])=>s+(+p-me[c-2]+1)*PUBLISHED[c],0)+paperDifficulty(active.map(([c])=>+c),7,1,6,5)>=29;
  if(name==='glenn2010_linear')return active.reduce((s,[c,p])=>s+(+p-me[c-2]+1)*GLENN10_PROGRESS[c],0)+paperDifficulty(active.map(([c])=>+c),11,-1,4,4)>=61;
  if(name==='couto'||name==='couto-ga-stop'){
    if(active.length<3&&g.owner.some(x=>x===-1))return false;
    const cols=active.map(([c])=>+c);let score=-cols.length*cols.reduce((s,c)=>s+c,0);
    // Match Python's CoutoAgent exactly: advanced-this-turn multiplied by
    // the secured position (which is 0 for an opponent-owned column, the
    // completed height + 1 for our own column, and the current position
    // otherwise).  Do not substitute the raw absolute position here.
    score+=active.reduce((s,[c,p])=>{
      const column=+c, position=+p;
      const secured=g.owner[column-2]===player ? LENGTHS[column]+1
        : g.owner[column-2]===1-player ? 0
        : position;
      return s+PUBLISHED[column]*(position-me[column-2])*secured;
    },0);
    const d=paperDifficulty(cols,7,1,6,5);return score+d*d+d>=29;
  }
   if(name==='mc'||name==='mc-full-game'||name==='mc-strong')return await mcStop(g,player,name);
  if(name==='vmerckle_monte'||name==='vmerckle_monte_uct')return vmerckleStop(g,player,name);
  return false;
}
function featuresForDQN(g,player,strategic=false,domain=false,moves=null){const s=g.snapshot(),me=s.permanent[player],opp=s.permanent[1-player],lengths=[3,5,7,9,11,13,11,9,7,5,3],f=[];for(let i=0;i<11;i++)f.push(me[i]/lengths[i]);for(let i=0;i<11;i++)f.push(opp[i]/lengths[i]);for(const c of COLUMNS)f.push((s.active[c]||0)/LENGTHS[c]);for(const c of COLUMNS)f.push(s.active[c]?1:0);for(const o of s.owner)f.push(o===player?1:o===1-player?-1:0);const completed=s.owner.filter(o=>o===player).length,oppCompleted=s.owner.filter(o=>o===1-player).length;f.push(completed/3,oppCompleted/3,Math.min(s.rolls/20,1));const bust=bustProbability(g),risk=Object.entries(s.active).reduce((x,[c,p])=>x+(p-me[c-2])/LENGTHS[c],0),reached=Object.entries(s.active).filter(([c,p])=>p===LENGTHS[c]).length;f.push(bust,risk);if(domain||strategic){const mp=me.reduce((x,v,i)=>x+v/lengths[i],0)/11,op=opp.reduce((x,v,i)=>x+v/lengths[i],0)/11;const rule=rule28Points({...g,active:g.active},player);f.push(Object.keys(s.active).length/3,reached/3,Math.min(completed+reached,3)/3,completed+reached>=3?1:0,mp-op,risk*(1-bust),Math.min(rule/40,1),oppCompleted>=2?1:0);if(strategic){for(let i=0;i<11;i++){const c=COLUMNS[i];f.push(Math.max(0,(s.active[c]??me[i])-me[i])/lengths[i])}const near=opp.reduce((x,v,i)=>x+(s.owner[i]===-1&&lengths[i]-v<=1?1:0),0);f.push(near/3)}}return f;}
function modelFeatures(g,player,policy,moves=null){const base=policy.payload.observation_size-(policy.payload.action_size===8?18:0);const strategic=base===80;const domain=base===68||strategic;const f=featuresForDQN(g,player,strategic,domain);if(moves){for(const move of sortedMoves(moves).slice(0,6))f.push(move[0]/12,(move[1]??0)/12,1);while(f.length<base+18)f.push(0);}if(f.length!==base+(moves?18:0))return null;return f;}
function chooseLearnedPair(g,moves){if(!ppo||ppo.payload.action_size!==8)return null;const f=modelFeatures(g,g.player,ppo,moves);return f?ppo.pairAction(f,moves):null;}
async function decide(name,g,player){if(name==='dqn'&&model){const f=modelFeatures(g,player,model);return f?model.action(f):heuristicAction({...g.snapshot(),permanent:[g.permanent[player],g.permanent[1-player]]});}if(name==='ppo'&&ppo){const f=modelFeatures(g,player,ppo);return f?ppo.action(f):heuristicAction({...g.snapshot(),permanent:[g.permanent[player],g.permanent[1-player]]});}if(name==='dqn'||name==='ppo')return heuristicAction({...g.snapshot(),permanent:[g.permanent[player],g.permanent[1-player]]});return await stopDecision(g,name,player)?'stop':'continue';}
let runGeneration=0;const aiTimers=new Set();
function scheduleAI(delay,generation=runGeneration){const timer=setTimeout(()=>{aiTimers.delete(timer);if(generation===runGeneration)aiTurn(generation);},delay);aiTimers.add(timer);}
function invalidateAI(){runGeneration++;for(const timer of aiTimers)clearTimeout(timer);aiTimers.clear();pendingVmerckle.clear();return runGeneration;}
async function aiTurn(generation=runGeneration){if(generation!==runGeneration||game.winner!==null||!isAI(game.player))return;const name=agents[game.player];if(game.phase==='roll'){if(!game.roll()){render();scheduleAI(350,generation);return}const move=await choosePair(game,game.moves,name);if(generation!==runGeneration)return;game.applyMove(move);render();scheduleAI(350,generation);return}if(game.phase==='decision'){const action=await decide(name,game,game.player);if(generation!==runGeneration)return;action==='stop'?game.stop():game.continue();render();if(game.phase!=='done')scheduleAI(350,generation);}}
function start(){const generation=invalidateAI();game.reset();render();if(isAI(0))scheduleAI(300,generation);}
$('apply-agents').onclick=()=>{agents=[$('agent-a').value,$('agent-b').value];start()};$('new-game').onclick=start;
$('roll').onclick=()=>{invalidateAI();if(game.roll())render();else{render();scheduleAI(300)}};$('stop').onclick=()=>{invalidateAI();game.stop();render();scheduleAI(300)};$('continue').onclick=()=>{invalidateAI();game.continue();render();scheduleAI(300)};
localizeUI(); setupSelectors();
BrowserDQN.load('models/dqn.json').then(m=>{model=m;dqnLoading=false;render()});
BrowserDQN.load('models/ppo.json').then(m=>{ppo=m;ppoLoading=false;updateModelOptions();render()});
render();
