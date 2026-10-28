const messages = {
  en: {
    subtitle: 'An RL opponent running locally in your browser',
    newGame: 'New game',
    loadingModel: 'Loading models…',
    playerA: 'Player A (first player)',
    playerB: 'Player B',
    applyAgents: 'Apply game settings',
    board: 'Board',
    neutral: 'Neutral markers: {count} / 3',
    roll: 'Roll dice',
    stop: 'Stop and bank',
    continue: 'Continue',
    modelInfo: 'Model and runtime information',
    gameOver: 'Game over',
    turn: "{player}'s turn",
    marker: "Player {player}'s neutral marker",
    dqnLoaded: 'DDQN: loaded weights',
    dqnMissing: 'DDQN: unavailable (heuristic fallback)',
    ppoLoaded: 'PPO: loaded weights',
    ppoMissing: 'PPO: unavailable',
    modelsStatus: 'Models — DDQN: {dqn}; PPO: {ppo}',
    loading: 'loading',
    loaded: 'loaded',
    unavailable: 'unavailable',
    ppoUnavailable: 'Trained PPO (not published)',
    yourTurn: 'Your turn.',
    aiThinking: 'AI thinking…',
    bust: 'Bust! Unbanked progress is lost.',
    winner: 'Player {player} wins!',
    'agent.human': 'Human',
    'agent.random': 'Random',
    'agent.rule28': 'Rule of 28',
    'agent.rule28-smart': 'Smart Rule of 28',
    'agent.glenn-aloi': 'Glenn & Aloi',
    'agent.glenn2009_constant': 'Glenn 2009 constant',
    'agent.glenn2010_linear': 'Glenn 2010 linear',
    'agent.couto': 'Couto',
    'agent.couto-ga-stop': 'Couto + GA pairing',
    'agent.rule28-ga-pure': 'Rule of 28 + GA pairing',
    'agent.mc': 'Monte Carlo',
    'agent.mc-full-game': 'Full-game Monte Carlo',
    'agent.mc-strong': 'MC strong',
    'agent.vmerckle_monte': 'vmerckle Monte Carlo',
    'agent.vmerckle_monte_uct': 'vmerckle root UCB1',
    'agent.dqn': 'Trained DDQN',
    'agent.ppo': 'Trained PPO'
  },
  zh: {
    subtitle: '在瀏覽器本地執行的 RL 對手',
    newGame: '新遊戲', loadingModel: '載入模型中…',
    playerA: '玩家 A（先手）', playerB: '玩家 B', applyAgents: '套用對局設定',
    board: '棋盤', neutral: '中立棋：{count} / 3', roll: '擲骰子',
    stop: '停止並結算', continue: '繼續', modelInfo: '模型與執行資訊',
    gameOver: '遊戲結束', turn: '{player} 的回合', marker: '玩家 {player} 的中立棋',
    dqnLoaded: 'DDQN：已載入權重', dqnMissing: 'DDQN：不可用（heuristic fallback）',
    ppoLoaded: 'PPO：已載入權重', ppoMissing: 'PPO：不可用',
    modelsStatus: '模型 — DDQN：{dqn}；PPO：{ppo}',
    loading: '載入中', loaded: '已載入', unavailable: '不可用', ppoUnavailable: '訓練好的 PPO（尚未發布）',
    yourTurn: '輪到你。', aiThinking: 'AI 思考中…', bust: '爆掉了！本回合未結算進度。',
    winner: '玩家 {player} 獲勝！',
    'agent.human': '玩家操作', 'agent.random': 'Random', 'agent.rule28': 'Rule of 28',
    'agent.rule28-smart': 'Smart Rule of 28', 'agent.glenn-aloi': 'Glenn & Aloi',
    'agent.glenn2009_constant': 'Glenn 2009 constant', 'agent.glenn2010_linear': 'Glenn 2010 linear',
    'agent.couto': 'Couto', 'agent.couto-ga-stop': 'Couto + GA pairing',
    'agent.rule28-ga-pure': 'Rule of 28 + GA pairing', 'agent.mc': 'Monte Carlo',
    'agent.mc-full-game': 'Full-game Monte Carlo', 'agent.mc-strong': 'MC strong',
    'agent.vmerckle_monte': 'vmerckle Monte Carlo', 'agent.vmerckle_monte_uct': 'vmerckle root UCB1',
    'agent.dqn': 'Trained DDQN', 'agent.ppo': 'Trained PPO'
  }
};

let locale = 'en';
export function detectLocale(languages = globalThis.navigator?.languages || [globalThis.navigator?.language || 'en']) {
  return languages.some(language => /^(zh[-_](tw|hant|hk|mo)|zh-hant)/i.test(language)) ? 'zh' : 'en';
}
export function setLocale(next) { locale = next === 'zh' ? 'zh' : 'en'; if (globalThis.document) document.documentElement.lang = locale === 'zh' ? 'zh-Hant' : 'en'; }
export function t(key, values = {}) {
  const text = messages[locale][key] ?? messages.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => values[name] ?? `{${name}}`);
}
