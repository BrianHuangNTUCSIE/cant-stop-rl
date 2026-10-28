export const COLUMNS=[2,3,4,5,6,7,8,9,10,11,12];
export const LENGTHS={2:3,3:5,4:7,5:9,6:11,7:13,8:11,9:9,10:7,11:5,12:3};
export const GA_WEIGHTS={2:7,3:0,4:2,5:0,6:4,7:3,8:4,9:0,10:2,11:0,12:7};
import {t} from './i18n.js';

function combinations(dice){
  // The three partitions of four dice into two pairs, matching rules.py.
  const [a,b,c,d]=dice;
  // Keep partition order.  The Python engine uses this order for published
  // policies whose tie breaking is deliberately first-match.  Only complete
  // two-column moves are sorted below, because Move.columns is canonical.
  const partitions=[[a+b,c+d],[a+c,b+d],[a+d,b+c]];
  return partitions.filter((cols,i,all)=>all.findIndex(other=>other.join()==cols.join())===i);
}

export function legalMoves(dice, game){
  const choices=[];
  const activeCount=Object.keys(game.active).length;
  const canStep=c=>game.owner[c-2]===-1 &&
    (game.active[c]!==undefined
      ? game.active[c]<LENGTHS[c]
      : activeCount<3 && game.permanent[game.player][c-2]<LENGTHS[c]);
  const add=cols=>{if(!choices.some(old=>old.join()==cols.join()))choices.push(cols);};
  for(const [x,y] of combinations(dice)){
    if(x===y){
      if(!canStep(x)) continue;
      const next=(game.active[x]??game.permanent[game.player][x-2])+1;
      add(next<LENGTHS[x] ? [x,x] : [x]);
      continue;
    }
    const canX=canStep(x), canY=canStep(y);
    if(canX&&canY){
      if(activeCount+Number(game.active[x]===undefined)+Number(game.active[y]===undefined)<=3) add([x,y].sort((a,b)=>a-b));
      else { add([x]); add([y]); }
    } else if(canX) add([x]);
    else if(canY) add([y]);
  }
  return choices;
}

export function bustProbability(game){
  let busts=0; for(let a=1;a<=6;a++)for(let b=1;b<=6;b++)for(let c=1;c<=6;c++)for(let d=1;d<=6;d++)
    if(!legalMoves([a,b,c,d],game).length)busts++;
  return busts/1296;
}

export class Game {
  constructor(){this.lengths=LENGTHS;this.reset();}
  reset(){this.permanent=[[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]];this.owner=Array(11).fill(-1);this.active={};this.player=0;this.rolls=0;this.phase='roll';this.dice=[];this.moves=[];this.winner=null;this.message=t('playerTurn',{player:1});}
  roll(){if(this.phase!=='roll'||this.winner!==null)throw new Error('roll() only allowed in the roll phase');this.dice=Array.from({length:4},()=>1+Math.floor(Math.random()*6));this.moves=legalMoves(this.dice,this);if(!this.moves.length){this.message=t('bust');this.active={};this.rolls=0;this.phase='roll';this.player=1-this.player;return false;}this.phase='pair';return true;}
  applyMove(cols){
    if(this.phase!=='pair'||!this.moves.some(move=>move.join()==cols.join()))throw new Error('Expected a legal move after rolling');
    const nextActive={...this.active};
    for(const c of cols)nextActive[c]=(nextActive[c]??this.permanent[this.player][c-2])+1;
    if(Object.keys(nextActive).length>3)throw new Error('Cannot use more than 3 neutral markers');
    this.active=nextActive;this.rolls++;this.moves=[];this.phase='decision';
  }
  continue(){if(this.phase!=='decision')throw new Error('continue() only allowed after a move');this.phase='roll';}
  stop(){if(this.phase!=='decision')throw new Error('stop() only allowed after a move');for(const [c,pos] of Object.entries(this.active)){this.permanent[this.player][c-2]=pos;if(pos===LENGTHS[c])this.owner[c-2]=this.player;}if(this.owner.filter(x=>x===this.player).length>=3){this.winner=this.player;this.phase='done';this.active={};this.moves=[];this.message=t('winner',{player:this.player+1});return;}this.active={};this.rolls=0;this.player=1-this.player;this.phase='roll';this.message=t('playerTurn',{player:this.player+1});}
  snapshot(){
    // Match Python's immutable agent-facing State: callers must not be able to
    // mutate the live game through a diagnostic/features snapshot.
    return {
      permanent:this.permanent.map(row=>row.slice()), owner:this.owner.slice(),
      active:{...this.active}, player:this.player, lengths:{...LENGTHS},
      weights:{...GA_WEIGHTS}, rolls:this.rolls
    };
  }
}

export function chooseGA(game,moves){let best=-Infinity,chosen=[];for(const m of moves){const newMarkers=new Set(m.filter(c=>game.active[c]===undefined));const score=m.reduce((s,c)=>s+GA_WEIGHTS[c],0)-6*newMarkers.size;if(score>best){best=score;chosen=[m]}else if(score===best)chosen.push(m)}return chosen[Math.floor(Math.random()*chosen.length)];}
