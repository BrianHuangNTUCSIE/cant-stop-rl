# Can't Stop 架構遷移指南

## 實作狀態（2026-10-26）

第一個 milestone 已建立可回退的基礎路徑：

- `cant_stop/transition.py` 固定 schema version 1，並統一 terminal
  `next_state` 為固定長度 zero vector。
- `cant_stop/backend.py` 與 `config.py` 定義 backend boundary 和可寫入
  config/checkpoint 的 runtime metadata。
- `rollout/` 提供 persistent Python multiprocessing actors、batched dispatch、
  worker failure propagation 與明確 shutdown。
- `learners/dqn_learner.py` 將 replay/optimizer/target update 與 environment、
  opponent 分離。
- `train_dqn.py` 保留 `--backend python` / `python-reference`，並新增
  `--backend python-multiprocess --num-workers N`。單 worker smoke test 的 raw
  episode records、evaluation records 和最終權重與 reference backend 完全一致。
- `scripts/benchmark_backends.py` 量測 primitive、environment、actor transport、
  Torch inference 與 optimization throughput，並輸出 deterministic checksum。
- MC rollout 的低風險優化已移除每個 simulated roll 的 `State` snapshot 建立，
  並保留 pairing、tie-breaking 與 RNG sequence；MC-16 純 simulator benchmark
  約由 103 提升至 130 episodes/sec，舊/新版 rollout parity 已驗證。
- 2026-10-26 以 300 episodes、seed 441 重新測試：Python reference 約
  1,700 episodes/sec；Python multiprocessing 約 345 episodes/sec（1--4 workers），
  8 workers 約 278 episodes/sec。各組 checksum 均為
  `5262:5272:16416:10`，因此 multiprocessing 目前只作 parity prototype，
  正式訓練仍應使用 `--backend python`，不應藉增加 workers 宣稱加速。
- 不平行執行不同 training seeds。`train_multi_seed.py --jobs` 保留為
  相容性參數，但目前只接受 `--jobs 1`，以確保每個正式實驗的 CPU
  throughput 和資源配置可單獨解釋。
- Counterfactual MC labels 可使用單一 training run 內的 persistent worker pool：
  `--counterfactual-workers N`。這只平行化同一個 seed 的 label 計算，不改變
  episode seed、label seed 或 transition 插入順序；預設為 0（維持同步 reference）。
- Phase 3 已開始：`native/cant_stop_rules.hpp`、`native/rules_cli.cpp`、
  `native/transition_cli.cpp` 與 `native/trajectory_cli.cpp` 提供 fixed-array
  C++ `legal_moves`、`advance_active`、`roll/apply/continue/stop` prototype；
  `tests/test_native_rules.py` 以 seeded random cases 和 40 條 explicit-dice
  trajectories 做 Python differential test。這仍是 optional parity harness，
  尚未接入 training/evaluation backend，也不應視為 native simulator 已完成。

尚未實作 native/C++ backend；必須先以 benchmark 確認 Python actor 的實際
bottleneck，再開始 Phase 3。`train_dqn.py` 的 validation/checkpoint/CSV
orchestration 仍留在 entry point，後續可再抽成獨立 runner，但不得因此改變
既有實驗 protocol。

本文件描述如何將目前的 Python simulator/training pipeline 遷移成：

```text
native simulator / actor workers
        ↓ batched transitions
Python learner / PyTorch
        ↓ policy weights
native or Python actor workers
```

目標是提升 simulator、heuristic agent、Monte Carlo rollout 與 RL data collection 的 throughput，同時保留既有實驗的可重現性與結果可比性。本文件是實作指南，不是新的研究設定；遷移期間不能任意改變遊戲規則、reward、feature layout、seed protocol 或 evaluation protocol。

## 1. 現況與主要問題

目前核心程式位於：

```text
cant_stop/model.py       # Move, State, feature encoding
cant_stop/rules.py       # legal moves, bust probability, scores
cant_stop/game.py        # mutable game state machine
cant_stop/agents.py      # heuristic and MC agents
cant_stop/rl_env.py      # learner-facing environment
cant_stop/evaluation.py  # game execution and evaluation
cant_stop/dqn.py         # PyTorch model, replay, optimization
scripts/train_dqn.py          # training orchestration
```

### 1.1 `--num-envs` 不是真正的平行化

`train_dqn.py` 交錯處理多個 `EpisodeSlot`，但所有 environment 仍在同一個 Python process/thread 執行。它只能讓神經網路取得較大的 inference batch，無法平行化 `Game.roll()`、`legal_moves()`、`State.features()`、heuristic、MC rollout 或 opponent turn simulation。

### 1.2 可能的主要 bottleneck

優先量測而不是假設 PyTorch 是主因：

1. `MonteCarloAgent.should_stop()` 的 nested rollout loop
2. 每個 decision 建立 `State`、`dict`、`Move` 與 feature list
3. 每個 environment step 的 Python method dispatch
4. Python list 到 Torch tensor 的反覆轉換
5. 小型 MLP 的大量小 batch inference/update

## 2. 不可破壞的 invariants

若 native backend 與 Python reference 不一致，應以 Python reference 和既有 tests 作為 oracle。

### 2.1 遊戲規則

- 兩位玩家
- columns 2–12
- column heights：`3,5,7,9,11,13,11,9,7,5,3`
- 一回合最多 3 個 active columns
- 若兩個 dice sums 同時合法，必須同時使用
- bust 只丟失本回合未 bank 的 progress
- column 只有在 stop/bank 後才會 close
- 任一玩家完成三個 columns 即獲勝

### 2.2 隨機性

分離 dice RNG、seat 0/1 policy RNG、training exploration RNG、replay sampling RNG。不可使用 shared global RNG；native worker 也必須接受明確 seed，不能依賴 process 啟動順序。

### 2.3 Feature layout

目前 feature 數量為：raw 58、enhanced 60、domain 68、strategic 80；learned pairing 另外加 18 個 pairing-candidate features。native backend 必須產生與 `State.features()` 完全相同的數值、順序與 normalization，不能只測 shape。

### 2.4 Evaluation protocol

保留 alternating seats、deterministic paired seeds、max-turn safety limit、raw per-game records，以及 paired bootstrap 或多 seed variability inference。不要同時改 simulator 和 evaluation protocol。

## 3. 目標架構

建議新增以下責任分層：

```text
cant_stop/
    model.py, rules.py, game.py, agents.py, evaluation.py  # Python reference
    backend.py                  # backend protocol/factory
    transition.py               # fixed transition schema
    config.py                   # validated experiment config

native/
    CMakeLists.txt
    include/cant_stop_native.hpp
    src/rules.cpp, simulator.cpp, bindings.cpp

rollout/
    actor.py                    # actor lifecycle
    actor_pool.py               # process management
    protocol.py                 # messages and transition batches
    shared_buffer.py            # optional shared memory transport

learners/
    dqn_learner.py              # PyTorch optimization loop
    replay.py                   # replay ownership
    training_runner.py          # high-level orchestration
```

### 3.1 三層責任

**Simulation layer** 負責 compact game state、dice、legal moves、transition、terminal condition、features、heuristic 與 MC rollout；不負責 replay、optimizer、checkpoint 或報告。

**Actor layer** 負責 episode、environment、transition collection、episode metrics 與 policy snapshot；不負責 gradient update、checkpoint 或 global validation。

**Learner layer** 負責 policy network、batched inference、replay、gradient update、target network、checkpoint 和 validation；不應直接操作 opponent 細節。

## 4. 固定 transition schema

在平行化或 native 化前，先建立固定 schema：

```text
state:               float32[observation_size]
action:              int64
reward:              float32
next_state:          float32[observation_size]
done:                bool
n_steps:             int32
counterfactual:      float32[2]
counterfactual_mask:  bool
episode_id:          int64
worker_id:           int32
step_id:             int32
```

`episode_id`、`worker_id`、`step_id` 可作 debugging metadata。batch API 一次傳多筆 transition，不要每筆 transition 呼叫 Python/C++ binding。`next_state=None` 統一轉成 zero vector，與目前 replay 行為一致；dtype 與 schema version 必須固定並寫入 checkpoint/config。

## 5. 分階段遷移計畫

### Phase 0：baseline 與 profiling

先執行：

```bash
cd cant_stop_baselines
python3 -m unittest discover -s tests -t . -v
```

新增 benchmark，測量 games/sec、turns/sec、decisions/sec、`legal_moves()`、`State.features()`、Rule28、MC decision、environment step、network inference、`optimize_batch()`。固定 seed、opponent、episode count，分別測 `rule28-smart`、低/正式 rollout 的 `mc`、DQN vs Rule28、DQN vs MC。

**完成條件：**既有 tests 全部通過；benchmark 可重複；知道 simulator、opponent、learner 各佔總時間比例。

### Phase 1：拆解 `train_dqn.py`

先不改規則，也不導入 C++。拆出：

```text
TrainingConfig
OpponentFactory
EpisodeRunner
TransitionCollector
Learner
ValidationRunner
CheckpointManager
MetricsWriter
```

`EpisodeRunner` 只執行 environment；`Learner` 只知道 observation/action/transition；`OpponentFactory` 只建立 opponent；validation 不嵌入 learner update loop；checkpoint/CSV I/O 不阻塞 actor。相同 command line configuration 必須仍能產生可比較結果。

### Phase 2：Python multiprocessing actor prototype

先用 Python process 驗證平行 actor，避免同時引入 C++ 與 multiprocessing。

```text
main process
├── Learner
├── ReplayBuffer
├── ActorProcess 0
├── ActorProcess 1
└── ActorProcess N
```

每個 actor 自己擁有 environment，執行完整 episode 或固定 steps，再以 Queue/pipe 傳回 transition batch。先使用簡單 Queue；只有 profile 顯示 IPC 是瓶頸時才導入 shared memory。

worker seed 不可依賴啟動順序，例如：

```text
worker_seed = base_seed + worker_id * LARGE_CONSTANT + episode_id
```

policy weights 必須是 immutable snapshot；不得讓 actor 讀取 learner 正在修改的 model object。

**完成條件：**worker crash 可偵測；shutdown 不遺失 transitions；可 checkpoint/resume；`num_workers=1` 可與舊版比較；`num_workers>1` 有實際 throughput 改善。

### Phase 3：native simulator

移植順序：

1. compact `GameState`
2. `legal_moves`
3. `advance_active`
4. roll/apply/stop/continue
5. feature extraction
6. heuristic policies
7. MC rollouts
8. batch episode execution

native state 應使用固定大小結構，而不是模擬 Python `dict`/dataclass。active columns 可用 bitmask 或固定 array；每次 move 不應 heap allocate。

可使用 pybind11 或等價 binding，但 API 必須以 batch 為主，例如：

```python
native.collect_episodes(
    seeds, policy_config, opponent_config, max_steps
) -> TransitionBatch
```

避免以下逐 action API：

```python
native.roll()
native.choose_move()
native.apply_move()
```

這會保留大量 Python/C++ crossing overhead。

heuristic policy 用 enum/config 選擇 `RANDOM`、`RULE28`、`RULE28_SMART`、`GLENN_ALOI`、`COUTO`、`MC`，不要把 Python callback 傳給 C++。

### Phase 4：native actor integration

native 通過 differential tests 後，才替換 Phase 2 的 Python actor。保留：

```text
--backend python-reference
--backend python-multiprocess
--backend native
```

native actor 至少提供兩種模式：

- **完整 heuristic episode：**C++ 一次執行完整 game，Python 接收 game records，供 baseline/evaluation/data generation 使用。
- **RL transition collection：**actor 執行 engine/opponent，learner 提供 action，actor 以 batch 方式回傳 transitions。

不要每個 decision 建立新的 native object 或 process。

## 6. Differential testing

對相同 game state、dice、move、seed，比較 Python 與 native 的 legal moves、next state、phase、winner、features、heuristic action、MC decision（RNG protocol 相同時）。

測試至少涵蓋 duplicate sums、one-slot/two-new-number、closed column、marker limit、bust、bank、completed column 與 terminal win。feature 最大絕對誤差應不超過 `1e-6`。

## 7. DQN learner 修改方向

`dqn_learner.py` 只接收 observations、legal masks、actions、rewards、next observations、dones，不應呼叫 `Game.roll()`、`legal_moves()` 或 `make_agent()`。

`optimize_batch()` 在 counterfactual loss 啟用時，應保存同一次 `online(states)` 結果：

```python
q_values = online(states)
predicted = q_values.gather(...)
q_gap = q_values[:, continue_index] - q_values[:, stop_index]
```

transition batch 儘量在 producer 端使用固定 dtype，避免每次 update 從 nested Python lists 建立 tensor。對小型 MLP 至少 benchmark Torch threads `1,2,4,8` 與 actor workers `1,2,4,8`，避免每個 worker 都啟動多個 Torch threads 造成 oversubscription。

## 8. Benchmark 與驗收標準

每個 backend 使用相同 workload：

```text
backend: python / python-multiprocess / native
opponent: rule28-smart / mc
episodes: fixed count
seed: fixed
features: raw / enhanced / domain
pairing: ga / learned
```

記錄 episodes/sec、transitions/sec、decisions/sec、CPU utilization、memory、learner updates/sec、wall-clock、result checksum。分開報 simulation-only、actor-to-learner、learner 與 end-to-end throughput。

不要只因 native baseline evaluation 變快就宣稱 training 整體加速。

## 9. 建議的 agent 分工

- **Profiling agent：**benchmark、phase timing、baseline report。
- **Training refactor agent：**拆 `train_dqn.py`，建立 learner/actor/collector interfaces。
- **Python parallel agent：**actor process pool、queue protocol、worker failure、seed、shutdown。
- **Native rules agent：**compact state、legal moves、game transitions、C++ tests。
- **Native policies agent：**Rule28、GA、Glenn-Aloi、Couto、MC rollout。
- **Integration agent：**native actor、CLI backend switch、metadata、end-to-end benchmark。

不同 agent 不應同時修改同一核心檔案。先完成 Phase 0/1 的 interface，再平行處理 Phase 2/3。

## 10. 常見錯誤

### 只把 heuristic function 綁到 C++

逐 move crossing boundary 往往收益有限；應把完整 episode 或 rollout batch 放在 native side。

### 共用 RNG

會造成不可重現或重複 game。每個 worker 必須有明確 seed stream。

### native backend 順便改規則

效能重構與規則修改必須分開。規則變動應另立 experiment 與 tests。

### 為速度刪除 raw game records

正式 evaluation 仍需保存資料；可以提供 benchmark-only 模式，但不能默默刪除 evidence。

### actor 直接修改 learner model

所有 weights 必須是 immutable snapshot 或明確 version，transition 應記錄 policy version。

### 刪除 Python reference

native backend 完成前，Python reference 是 correctness oracle，不應刪除。

## 11. 第一個 implementation milestone

第一個 milestone 不應是完整 C++ rewrite，而應是：

1. 新增 benchmark
2. 將 `train_dqn.py` 拆成 learner/actor interfaces
3. 使用 Python multiprocessing actors
4. 保留 `--backend python`
5. 驗證 `num_workers=1` 與現有版本一致
6. 量測 `num_workers=1,2,4,8`

若 Python multiprocessing 已使 environment throughput 提升到 learner 成為瓶頸，下一步才優化 Torch batch 或引入 native simulator。

## 12. Rollback 與 metadata

每一階段都必須能回到上一 backend，不要直接替換現有 `Game`、`Agent` 或 `evaluate()`；先用 adapter 接入。

正式 experiment config 應記錄：

```json
{
  "backend": "python-reference",
  "worker_count": 1,
  "policy_version": 1,
  "transition_schema_version": 1,
  "seed_formula": "...",
  "native_commit": null
}
```

如此才能區分模型改進、simulator 改進、平行化 throughput 改進，以及 backend 差異。
