# Can't Stop AI — Research-to-Implementation Specification

> **Audience:** Local coding agent (OpenCode / Claude Code / Codex) working inside an existing Can't Stop project.
> **Objective:** Integrate independently selectable, reproducible AI opponents into the **existing** game engine, with consistent rules, interfaces, evaluation, and provenance. Do not replace the engine or overwrite the existing DDQN.
> **Language:** This document is written in Traditional Chinese; keep code, API names, and commit messages in English.
> **Research scope:** Glenn & Aloi (2009), Glenn (2010), Medeiros et al. (2022), Lanctot et al. (2013), Carter (2026), and relevant public GitHub implementations.

## 0. 非協商原則（先讀）

1. **Inspect before coding.** 先閱讀 `AGENTS.md`、教授的 project spec、現有遊戲規則、agent API、baseline、DDQN、evaluator 與 tests。不得假設本文件的介面就是 repo 的介面。
2. **Preserve the game engine.** 使用薄 adapter 整合新 agent；不得為了移植演算法另造一套遊戲規則。若參考實作與現有規則不同，明確記錄並以本專案規則為準。
3. **Do not fake reproduction.** 只有在程式、權重、超參數、decision logic 足以一致重建時才能標記 `faithful_reproduction`。否則標記 `adaptation` 或 `inspired_baseline`。
4. **Separate two decisions.** Can't Stop 有 (a) 骰子配對／選 column，(b) 繼續擲骰／停止；研究常只解決其中一個。未定義的部分必須使用明確命名的既有策略，不可默默補上。
5. **No training by default.** 優先移植已公布的 champion、推論程式或可直接運行的演算法。訓練／synthesis 應是可選的獨立命令，不可讓測試自動觸發長時間訓練。
6. **No invented results.** 不可宣稱論文勝率等於在本 engine 的勝率；每個比較都要在同一規則、同一評估器下重新測量。
7. **License discipline.** 公開可讀的 GitHub repo 不代表允許複製。先查 `LICENSE`、NOTICE、來源註解。無明確授權時，優先獨立重實作論文中公開的演算法；不要直接複製原始碼。保留 attribution。
8. **Keep existing baselines.** `random`、`rule28-smart`、`mc`、`ddqn` 應繼續可用，且新增功能不應改變它們的預設行為。
9. **Research uncertainty is a feature.** 任何無法從論文或來源核實的細節應寫入 `docs/reproduction_gaps.md`，不可猜測後宣稱 exact。

## 1. 實作範圍與優先順序

### 分級

- **A — source-backed:** 有可公開檢視的實作；可讀、可測、可依授權移植。若無授權，僅作為理解行為的參考。
- **B — paper-backed:** 論文公開足夠公式、參數或 pseudocode，可獨立重建核心方法；仍可能有規則／tie-break 等差異。
- **C — partial:** 研究方向有價值，但無法核實完整決策函式、模型權重或完整訓練流程；不得標記為 exact。

| ID | 演算法／策略 | 主要類型 | 來源級別 | 預設優先級 | 交付標籤 |
|---|---|---|---|---|---|
| `glenn2009_constant` | GA-optimized Constant heuristic | Rule-based | B | P0 | `paper_reimplementation` |
| `glenn2010_linear` | GA-optimized Linear champion | Rule-based | B | P0 | `paper_reimplementation` |
| `sketch_ga_champion` | 2022 paper 使用的 GA champion（如原 repo 提供） | Programmatic | A（待查） | P1 | `source_port` 或 `unavailable` |
| `sketch_sa` | Sketch-guided Simulated Annealing synthesis | Program synthesis | A | P1 | `source_port`／`adaptation` |
| `sketch_uct` | Sketch-guided UCT synthesis | Program synthesis | A | P2 | `source_port`／`adaptation` |
| `vmerckle_monte` | Monte Carlo rollout decision | Search | A（授權待查） | P1 | `clean_room_reimplementation` |
| `vmerckle_monte_uct` | Root-level UCB / bandit search | Search | A（授權待查） | P1 | `clean_room_reimplementation` |
| `expectimax` | Chance-node expectimax | Search | B（generic） | P2 | `adaptation` |
| `star1ss` / `star2ss` | MCMS sampled *-minimax | Search | B（部分細節待查） | P2 | `adaptation` |
| `mcts_dpw` | MCTS with double progressive widening | Search | B（部分細節待查） | P2 | `adaptation` |
| `carter2026_dqn` | DQN stop/continue, heuristic dice selection | RL | A + thesis | P2 | `architecture_reproduction`（除非取得原權重） |
| `lourenco_polynomial` | Probability + polynomial heuristic | Rule-based | A（授權待查） | P3 | `clean_room_reimplementation` |
| `tabletopgames_mcts` | Generic framework search reference | Framework | A | P3 | `reference_only` |

**注意：** `sketch_sa` / `sketch_uct` 原本是**合成策略的程序**，不是單一固定策略。應分別提供 synthesis pipeline 與其輸出之可執行 policy。`carter2026_dqn` 若只有模型架構、沒有原作者 checkpoint，不能聲稱複製了原作者的 trained agent。

## 2. 必讀資料與連結（讀取順序）

### 2.1 Primary research

1. **Glenn & Aloi (2009), _A Generalized Heuristic for Can't Stop_** — https://cdn.aaai.org/ocs/123/123-2430-1-PB.pdf
   - 查閱：評分函式、column progress、pair choice、risk/stop decision、Constant/Linear 參數表、Rule of 28 比較。
   - **必須從原 PDF 逐式轉錄 scoring equation**；下面的權重表只是核對清單，不能替代公式。
2. **Glenn (2010), _Optimizing Genetic Algorithm Parameters for a Stochastic Game_** — https://www.scitepress.org/PublishedPapers/2010/30791/30791.pdf
   - 查閱：修正後的 Linear champion、完整參數、position normalization、rounding/floor、stop threshold、與 2009 版本差異。
3. **Medeiros et al. (2022), _What Can We Learn Even from the Weakest? Learning Sketches for Programmatic Strategies_** — local PDF: `04719-MedeirosL(2).pdf`（若存在於 repo；否則請使用使用者提供之附件）；code: https://github.com/leandrocouto/sketch-learning
   - Read repo README: https://github.com/leandrocouto/sketch-learning/blob/master/cant-stop-sketch-learning/README.md
   - 查閱：DSL、human/Glenn/random demonstrations、sketch learning、SA、UCT、evaluation protocol。
4. **Lanctot et al. (2013), _Monte Carlo *-Minimax Search_** — https://www.ijcai.org/Proceedings/13/Papers/093.pdf
   - 查閱：Star1/Star2、ExpSS、Star1SS、Star2SS、chance sampling、MCTS-DPW、evaluation function 與 game-specific settings 是否完全揭露。
5. **J. Cameron Carter (2026), _Using Machine Learning to Optimize the Board Game Can't Stop_** — local PDF: `Using Machine Learning to Optimize The Board Game Cant Stop(2).pdf`（若存在）；source: https://github.com/JCameronCarter/HonorsThesis
   - 查閱：51-feature encoding、stop/continue-only DQN、dice heuristic、network、reward、training、Appendix B code。

### 2.2 Public source / comparison projects

- **Sketch learning** — https://github.com/leandrocouto/sketch-learning
- **Carter DQN** — https://github.com/JCameronCarter/HonorsThesis
- **C++ Monte Carlo + UCB** — https://github.com/vmerckle/CantStop-AI ; main file https://github.com/vmerckle/CantStop-AI/blob/master/monte.cpp
- **C# probability / polynomial agent** — https://github.com/RafaelNLourenco/Autonomous-system
- **Tabletop Games framework (MIT; rules and generic search)** — https://github.com/GAIGResearch/TabletopGames ; https://tabletopgames.ai/wiki/games/implemented/
- **Python engine reference (MIT; not an AI strategy)** — https://github.com/recommend-games/cant-stop

**Do not automatically clone and copy code.** First verify availability, repository layout, commit SHA, license, dependencies and whether the code really contains the claimed strategy. Record findings in `docs/source_audit.md`.

## 3. Existing-engine audit — 必須先完成

Find and document:

- Rules: players, 3 columns to win, 4 dice and 3 pairings, active marker count, column heights (2–12), temporary/permanent progress, completed/blocked columns, bust resolution, turn end, victory.
- State shape: current player, all player positions, temporary positions, active columns, finished columns, legal dice choices, whether STOP is legal, terminal result.
- Decision points: pair selection vs continue/stop; who owns the RNG; whether the engine exposes chance outcomes or a clone/simulate API.
- Current baselines and their configuration; DDQN input encoding, action mask, model loading and evaluator.
- Existing test harness and game logs; exact seed semantics and alternating first-player policy.

Deliver `docs/engine_integration_audit.md` with actual file paths, discovered signatures and any incompatibilities. If engine has a single action-selection callback, implement a bridge rather than editing all agents.

### Suggested conceptual adapter (adapt names to actual repo)

```python
from typing import Protocol, Sequence, Any

class CantStopAgent(Protocol):
    def choose_pair(self, observation: Any, legal_pairs: Sequence[Any], rng: Any) -> Any:
        ...
    def choose_continue(self, observation: Any, rng: Any) -> bool:
        """True = roll again; False = bank/stop."""
        ...

class SearchContext(Protocol):
    def clone(self) -> Any: ...
    def legal_actions(self, state: Any) -> Sequence[Any]: ...
    def apply(self, state: Any, action: Any) -> Any: ...
    def sample_chance(self, state: Any, rng: Any) -> Any: ...
    def terminal_utility(self, state: Any, player_id: int) -> float: ...
```

This is a conceptual contract, **not** an instruction to overwrite the existing public API. Agents should be registered through the project's current factory/registry. Make configuration explicit, e.g. `--agent glenn2010_linear --seed 441`.

## 4. P0: Glenn heuristic champions

### 4.1 Parameter checklist — 2009 Constant

**Verify against original paper before committing.** Values below are transcription targets, not a complete algorithm.

| Symmetric columns | Progress weight | Move weight |
|---|---:|---:|
| 2,12 | 7 | 7 |
| 3,11 | 7 | 0 |
| 4,10 | 3 | 2 |
| 5,9 | 2 | 0 |
| 6,8 | 2 | 4 |
| 7 | 1 | 3 |

Other parameters to verify: `odds=7`, `evens=1`, `highs=6`, `lows=5`, `marker_penalty=6`, `stop_threshold=29`.

### 4.2 Parameter checklist — 2010 Linear champion

**Verify exact definition of `x` and whether each formula is used as a progress or move coefficient.** Do not assume normalization without reading the paper.

| Symmetric columns | Progress weight | Move weight formula |
|---|---:|---|
| 2,12 | 15 | `floor(72*x + 15)` |
| 3,11 | 13 | `floor(48*x)` |
| 4,10 | 8 | `floor(56*x + 3)` |
| 5,9 | 8 | `floor(44*x)` |
| 6,8 | 4 | `floor(28*x + 7)` |
| 7 | 3 | `floor(32*x + 5)` |

Other parameters to verify: `odds=11`, `evens=-1`, `highs=4`, `lows=4`, `marker_penalty=12`, `stop_threshold=61`.

### 4.3 Required behavior

- Reproduce the paper's **complete** score and stop logic, including symmetry, progress denominator, marker penalties, bonuses, tie breaking, invalid pairs, bust and blocked columns.
- Define a deterministic tie-break (or seeded randomized tie-break if paper specifies it); disclose deviations.
- If paper rules differ from current engine, preserve engine rules and mark `adaptation`, with a targeted test.
- Add unit tests using hand-computed score examples from original equations, boundary positions and all legal dice pairings.
- Do **not** retrain GA as a prerequisite: these are published champion weights.

## 5. P1: Medeiros 2022 programmatic synthesis

### 5.1 Required reconnaissance

Inspect original repository before implementing. Locate the actual DSL grammar, interpreter, AST representation, SA neighborhood operators, UCT search, sketch-learning procedure, demonstration formats, GA champion, and evaluation code. Determine whether the output strategies are stored as programs, serialized trees, or generated files.

### 5.2 Integration deliverables

1. `sketch_ga_champion`: If an executable GA champion is actually present, port or reimplement it; otherwise mark `unavailable` with evidence.
2. `sketch_sa`: Implement the **synthesizer** as an optional offline command; persist synthesized policy and metadata; add an inference adapter that executes the saved program in the existing engine.
3. `sketch_uct`: Same as SA, but optional P2 if computationally expensive.
4. `dsl_interpreter`: Match original grammar and program semantics where possible. Preserve short-circuiting, numeric types, comparisons, legal-action handling and randomness.
5. `demonstration_adapter`: Only if needed to learn sketches; translate original training records to existing observations without leaking future information.

### 5.3 Strict validity rules

- A synthesized strategy is **not** identical to the paper's winning strategy merely because the same synthesizer was run.
- Save random seed, source revision, demonstration dataset, search budget, objective, DSL version and synthesized AST/text.
- If original demonstration files are absent, do not invent replacements and claim reproduction. Use `adaptation` and describe substitute data.
- The original experiment's large search budget may be expensive. Provide `--budget-small` smoke test and `--budget-paper` separate configuration; do not silently run days of computation.

## 6. P1: Monte Carlo and root-UCB search

### 6.1 `vmerckle_monte`

Read `monte.cpp`; identify the exact sampling, rollout policy, terminal reward, stopping rule and action budget. Implement a version on the current engine, with explicit `num_rollouts` and RNG seed. Verify whether the original code selects dice pairs, decides stop/continue, or both.

### 6.2 `vmerckle_monte_uct`

Read source to identify whether it is root-only UCB1 selection or full tree UCT. **Do not label a root-bandit implementation as full MCTS.** Preserve the original exploration constant if present, otherwise mark adaptation. Use the same rollout policy and computational budget across comparisons.

### 6.3 Search implementation tests

- Zero/one rollout handling; all-legal-actions coverage; deterministic seeded results.
- Correct simulation of bust, banking, victory and opponent turns.
- Search must not mutate real game state or consume global game RNG.
- Check whether performance improves plausibly with increasing budget; do not assert monotonic improvement on each seed.

## 7. P2: MCMS, Expectimax, MCTS-DPW

Paper: https://www.ijcai.org/Proceedings/13/Papers/093.pdf

Implement separately, **only after** verifying the paper's exact algorithm pseudocode and what game-specific information is missing.

- `expectimax`: deterministic player-choice nodes and probabilistic dice/chance nodes; enumerate legal outcomes or use documented sampling approximation.
- `star1ss` / `star2ss`: use sampled chance outcomes and the paper's bound / pruning logic. Do not substitute ordinary expectimax and rename it Star2SS.
- `mcts_dpw`: implement full tree search with double progressive widening if justified by actual chance/action branching; expose budget and widening parameters.
- Parameter targets **to verify from the paper**: ExpSS chance sample `25`; Star1SS `30`; Star2SS `15`; MCTS-DPW `C1=50`, `dr=10`, `C2=25`, `alpha=0.3`; paper used roughly `200 ms` decision budgets. Confirm exact meaning and context of each parameter.

**Reproduction limit:** If the original Can't Stop evaluation function, exact transition ordering or rollout policy is not published, label `adaptation`. A faithful implementation of a *general algorithm* is not necessarily an exact reproduction of the *published Can't Stop agent*.

## 8. P2: Carter 2026 DQN

Sources: local thesis PDF (Appendix B) and https://github.com/JCameronCarter/HonorsThesis

### Architecture checklist (verify in thesis/source)

- Decision: **continue vs stop only**; dice pair selection is separate heuristic.
- Observation: reported `51` features; reproduce **exact ordering, scaling and player perspective** from code, not just dimensionality.
- Network: reported MLP `51 → 64 → 64 → 2` with ReLU hidden layers.
- Training examples: replay buffer about `1000`, minibatch `16`, learning rate `1e-4`; discount factor was modified across experiments (`0.95`, `0.5`, `0.3`). These are experiment-specific, **not** a single canonical configuration.
- The paper discusses Q-value overestimation; do not silently upgrade it to Double DQN and call it Carter's model.

### Required variants

- `carter2026_dqn_original_checkpoint` **only if** an authentic published checkpoint can be found and loaded; otherwise do not register this name.
- `carter2026_dqn_retrained` with explicitly documented dataset, seed, reward, episodes, hyperparameters and code differences.
- Keep `ddqn` as a separate existing agent; add a comparison with **identical dice-pair heuristic** where feasible to isolate stop-policy quality.

**Do not report reproduced published win rates without running evaluations.** Distinguish architecture replication, training replication and checkpoint replication.

## 9. P3: Other codebases

- `RafaelNLourenco/Autonomous-system`: Inspect C# code for probability estimation and first-degree polynomial pair selection; independently reimplement if licensing requires. Do not accept a repository's competition-champion claim without independent corroboration.
- `GAIGResearch/TabletopGames`: Use as independent rules oracle / generic search reference when compatible, not as proof of a published Can't Stop-trained agent. MIT license still requires attribution.
- `recommend-games/cant-stop`: Use as an optional cross-engine test oracle; **not** a competitive agent.
- Older/unverified work (e.g. Cheong 2020, Canakci et al. 2016, Yang 2024, simplified retrograde studies): catalog under `docs/unreproduced_references.md` only unless complete implementation details become available.

## 10. Evaluation protocol — mandatory

### 10.1 Match design

- Run all agents in **the same engine** and under identical rules.
- Alternate starting player and, where practical, use paired seeds / common random numbers without coupling agent-private RNG.
- Report agent win rate, draws if possible, number of games, confidence interval (e.g. Wilson 95%), seed list, budget, and mean decision latency.
- Separate **solitaire expected turns** from **multiplayer head-to-head win rate**. They measure different objectives.
- For search agents, report both **equal decision-time** and **equal rollout-budget** comparisons where meaningful.
- Fix opponent mix and pair-selection heuristic for targeted ablations; avoid conflating policy improvements with different dice heuristics.
- No selective reporting of the best seed; report mean/variance over independent seeds when training is involved.

### 10.2 Required comparison matrix

Core: `random`, `rule28-smart`, existing `mc`, existing `ddqn`, `glenn2009_constant`, `glenn2010_linear`, `vmerckle_monte`, `vmerckle_monte_uct`. Add `sketch_ga_champion`, saved Sketch-SA policies, Carter DQN and MCMS/MCTS agents only once available and validated.

For each new agent, evaluate at least against `rule28-smart`, existing `mc`, and existing `ddqn`; optionally run full round-robin. Begin with 100–500-game smoke tests, then use a statistically defensible final sample (e.g. 5,000–10,000 matches per pairing if computationally affordable), and report actual uncertainty.

### 10.3 Example command-line behavior (adapt to repo)

```bash
python -m project.evaluate --agent-a glenn2010_linear --agent-b ddqn --games 10000 --seed 441 --alternate-start --out results/glenn2010_vs_ddqn.json
python -m project.evaluate --round-robin configs/agents.yaml --games-per-pair 5000 --out results/round_robin.csv
python -m project.train_sketch --method sa --budget-small --seed 441 --output checkpoints/sketch_sa_seed441.json
```

These commands are **illustrative**. Implement equivalents using the existing CLI conventions.

## 11. Suggested repository changes (adapt to actual structure)

```text
agents/
  glenn2009_constant.py
  glenn2010_linear.py
  monte_rollout.py
  monte_root_ucb.py
  sketch_program.py
  carter_dqn.py
  search/
    expectimax.py
    mcms.py
    mcts_dpw.py
configs/
  reproducible_agents.yaml
  search_budgets.yaml
docs/
  engine_integration_audit.md
  source_audit.md
  reproduction_gaps.md
  algorithm_provenance.md
  unreproduced_references.md
  experiment_protocol.md
results/
  README.md
tests/
  test_glenn_heuristics.py
  test_agent_legality.py
  test_agent_determinism.py
  test_search_simulation.py
  test_integration_match.py
```

Do not create duplicate directories or APIs if the repo already has equivalents.

## 12. Acceptance criteria

Each agent is complete only when:

- [ ] Agent is registered and selectable without changing core engine logic.
- [ ] Every action is legal for the observed state; no hidden opponent information is read.
- [ ] Tests cover bust, stop/bank, terminal win, blocked columns, all 3 pairings and tie cases.
- [ ] Seeded inference is reproducible (within documented numerical/platform limits).
- [ ] No mutation of real state during search; no global RNG contamination.
- [ ] Source URL, paper citation, commit SHA, license status, algorithm classification and deviations are documented.
- [ ] A smoke evaluation vs existing baseline runs successfully.
- [ ] Results are saved with game count, seeds, start-order policy, opponent, version and compute budget.
- [ ] Agent's `faithful_reproduction` / `adaptation` / `inspired_baseline` classification is justified.
- [ ] README includes usage and performance caveats.

## 13. Agent execution plan — follow this order

**Phase 0 — audit only:** Read the project and primary papers; generate `docs/engine_integration_audit.md`, `docs/source_audit.md`, and an implementation checklist. Check licenses before porting. If paper tables/formulas are ambiguous, stop and log exact missing details.

**Phase 1 — deterministic heuristics:** Implement and test Glenn 2009 Constant and Glenn 2010 Linear; confirm pair-choice and stop logic from the actual papers; integrate registry and run smoke comparisons.

**Phase 2 — sampling search:** Implement the source-informed Monte Carlo and root-UCB agents. Reuse the engine simulator; test legality, seed reproducibility and performance across budgets.

**Phase 3 — programmatic strategies:** Inspect and integrate the Medeiros GA champion if present; implement DSL interpreter, offline SA/UCT synthesis and loadable saved policies. Do not block other phases on expensive synthesis.

**Phase 4 — DQN / deeper search:** Integrate Carter architecture and training/checkpoint handling; then implement clearly labeled Expectimax, MCMS and MCTS-DPW adaptations if time permits.

**Phase 5 — evaluation:** Run common protocol and produce `results/summary.md` with raw tables, uncertainty, costs and provenance. Highlight whether each agent materially beats the existing MC and DDQN; do not infer significance from point estimates alone.

### First action to take now

> Inspect the current repository and `AGENTS.md`. Then report the actual game/agent/evaluator APIs and a concrete file-by-file implementation plan. **Do not begin large-scale training or rewrite the engine.** Implement Phase 1 once compatibility and exact paper formulas are confirmed.

## 14. Source integrity and open questions

This document is a research-based implementation brief, **not** an assertion that every paper's exact source, trained weights, and full experimental setup are obtainable. The local agent must verify primary sources and source code, especially:

1. Exact Glenn 2009/2010 scoring equations and the interpretation of the weights and `x`.
2. Whether the Sketch repo includes a directly runnable GA champion, serialized synthesized programs, and all demonstration data.
3. Whether vmerckle's `monte.cpp` uses root UCB or a deeper UCT tree, and its exact rollout behavior.
4. Whether Carter publishes a genuine model checkpoint, and whether the 51-feature order is fully specified.
5. Whether MCMS paper publishes sufficient Can't Stop-specific evaluation and rollout details for a true replication.
6. Licenses for all GitHub sources before copying code.

When uncertain, preserve the original baseline and document the uncertainty rather than silently inventing missing implementation details.
