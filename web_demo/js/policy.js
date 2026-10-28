// Pure policy helpers shared by the browser implementation and parity tests.
export const GA_WEIGHTS={2:7,3:0,4:2,5:0,6:4,7:3,8:4,9:0,10:2,11:0,12:7};

export function gaPairScore(active, move) {
  const newMarkers=new Set(move.filter(column=>active[column]===undefined));
  return move.reduce((score,column)=>score+GA_WEIGHTS[column],0)-6*newMarkers.size;
}

export function sortedMoves(moves) {
  return moves.slice().sort((a,b)=>a[0]-b[0]||((a[1]??0)-(b[1]??0)));
}
