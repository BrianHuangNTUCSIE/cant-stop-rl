# Can't Stop Browser Demo

這是一個不需要後端的純前端 demo。介面可選擇雙方 agent，包括專案中的 DDQN、Rule28、Glenn & Aloi、Glenn 2009/2010、Couto、Random，以及瀏覽器版的 Monte Carlo / root-UCB 適配。直接用瀏覽器開啟 `index.html`，或在本資料夾執行：

```bash
python3 -m http.server 8000
```

然後開啟 <http://localhost:8000>。使用 HTTP server 是因為瀏覽器通常不允許 `file://` 載入 ES modules 與 JSON。

## 安裝訓練好的 DQN

目前 demo 沒有放入 binary `.pt` 檔。請在原專案已安裝 PyTorch 的環境執行：

```bash
../.venv/bin/python tools/export_weights.py \
  ../cant_stop_baselines/runs/ddqn_mc_counterfactual_seed447/best.pt \
  models/dqn.json
```

（若從原專案的 `final` 目錄執行，請自行調整 checkpoint 路徑。）產生 `models/dqn.json` 後，網頁會自動使用 DDQN；沒有模型時會顯示並使用 heuristic fallback。PPO checkpoint 也可使用同一 exporter，輸出成 `models/ppo.json`。

### Agent fidelity

遊戲規則、合法走法順序、Rule 28、Glenn & Aloi、Glenn 2009
Constant、Glenn 2010 Linear，以及 Couto 的 pairing/Stop 判斷會對齊目前
Python engine；paper agent 的平手處理也保留 Python 的 first-max 行為。
Random 與 Python 一樣是隨機 policy。`mc`、`mc-full-game`、`mc-strong`、
`vmerckle_monte` 和 `vmerckle_monte_uct` 也都有對應的 root/action search；
瀏覽器版本使用 JavaScript 的非 seeded RNG，且為保持互動性採用較小的 rollout
budget（正式報告中的結果仍應使用原 Python simulator）。

Browser 的 `snapshot()` 會回傳複製資料，agent 或 debug code 不會透過
snapshot 修改正在進行的遊戲。PPO exporter 會匯出 policy head（而非 value
head），因此產生的 `models/ppo.json` 可直接在瀏覽器執行。
八動作 PPO checkpoint 會在 pair phase 使用 logits 0--5 選擇排序後的合法配對，並在
decision phase 使用 logits 6--7；兩動作 PPO 維持 GA pairing。

介面中的玩家 A、玩家 B 都可以各自選擇 agent；選擇 `玩家操作` 可以進行本地雙人遊戲，選擇兩個 AI 則會自動播放整局。

## 架構

- `js/game.js`: 瀏覽器遊戲狀態機與 GA dice pairing
- `js/model.js`: dependency-free MLP inference
- `js/app.js`: UI 與 AI 回合流程
- `tools/export_weights.py`: PyTorch checkpoint 到瀏覽器 JSON 的轉換

模型權重會公開給使用者，這是純前端應用的正常限制；不適合用於需要保護模型或防作弊的遊戲。
