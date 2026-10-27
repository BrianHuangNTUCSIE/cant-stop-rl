# Research and Source Audit

**Audit date:** 2026-10-27  
**Method:** primary PDFs were downloaded and text-extracted for equation/table checks; public repositories were inspected at the SHAs below. No source code has been copied in Phase 0.

## Primary papers

| Source | Verification | Relevant result / status |
|---|---|---|
| Glenn & Aloi (2009), *A Generalized Heuristic for Can't Stop* | PDF available at https://cdn.aaai.org/ocs/123/123-2430-1-PB.pdf | Table 1 confirms Constant champion: progress `(7,7,3,2,2,1)`, move `(7,0,2,0,4,3)`, odds `7`, evens `1`, highs `6`, lows `5`, marker `6`, threshold `29`. Equation defines progress as `sum_i (s_i+1)(|7-i|+1)` and generalized move value as the sum of space weights from `m_i+1` through `n_i`. Verified. |
| Glenn (2010), *Optimizing Genetic Algorithm Parameters for a Stochastic Game* | PDF available at https://www.scitepress.org/PublishedPapers/2010/30791/30791.pdf | Table 6 confirms the later Linear champion: progress `(7,13,8,8,4,3)`, move functions `(72x+15,48x,56x+3,44x,28x+7,32x+5)`, odds `11`, evens `-1`, highs/lows `4`, marker `12`, threshold `61`. Equation (4) uses `floor(m_i * j/l_i + b_i)`. This conflicts with the guide’s Section 4.2 transcription; Phase 1 is blocked until the intended variant is explicitly chosen. |
| Medeiros et al. (2022), *What Can We Learn Even from the Weakest?* | Public repository README and layout verified at https://github.com/leandrocouto/sketch-learning, HEAD `dfcedeba7d5401fef4ef555ea9b4b39cf85a8773` | Repository contains `cant-stop-sketch-learning`, source, results, and `human_matches.pkl`, `glenn_matches.pkl`, `random_matches.pkl`; README documents SA/UCT/sketch modes and MIT license. A runnable GA champion is not established by the README alone; inspect the subdirectory and files before porting. |
| Lanctot et al. (2013), *Monte Carlo \*-Minimax Search* | PDF available at https://www.ijcai.org/Proceedings/13/Papers/093.pdf | Paper covers ExpSS, Star1/Star2, Star1SS/Star2SS, and MCTS-DPW, including Can't Stop evaluation. It reports a 200 ms search limit and game-specific settings, but exact reusable engine transition ordering, rollout policy, and evaluation details are not sufficient to call a future port a faithful reproduction without further extraction. |
| Carter (2026), *Using Machine Learning to Optimize the Board Game Can't Stop* | Public repository https://github.com/JCameronCarter/HonorsThesis, HEAD `139a8dbaba50a0c2735e8a518f1f1fe3ee4d5657` | Repository contains `thesis.pdf`, `Code/`, `Data/`, and `Code/dqn_multi_opp.keras` (SHA `e225a7571b1b70d2168272b21e6fed7d70c10a85`). `rl_player.py` verifies a 51-feature encoder, 64-64 ReLU/Dropout network, and action 0=stop/action 1=continue. The local thesis PDF named by the guide is absent; the public checkpoint is not redistributed or registered by default. |

## Public code sources and licenses

| Repository | HEAD checked | Observed license / use decision |
|---|---|---|
| `leandrocouto/sketch-learning` | `dfcedeba7d5401fef4ef555ea9b4b39cf85a8773` | MIT stated in README. Existing project README claims a ported Sketch-SA(O) notice, but the referenced local notice file is absent. Verify local provenance before copying or extending it. |
| `vmerckle/CantStop-AI` | `25be22910e3bfc5c351f38ff5461602d95dacd0a` | Public C++ repo contains `monte.cpp`, `ai.*`, and game files; no LICENSE file was found at the repository root. Treat as reference-only; independently reimplement until explicit licensing is confirmed. `monte.cpp` shows flat action sampling over pair/stop choices and a root UCB1 bandit, not full tree UCT. |
| `RafaelNLourenco/Autonomous-system` | `3d1a877efe43670dcdde9d86162b85ab33587854` | README documents probability-based Stop/Continue and first-degree-polynomial pair selection. No LICENSE file was found at the repository root. Reference-only / clean-room reimplementation. |
| `GAIGResearch/TabletopGames` | `033232baa88bac4b6a1fe14fae62939193cdf9f7` | MIT license visible in repository. Generic framework/rules reference only; not evidence of a Can't Stop-trained strategy. |
| `recommend-games/cant-stop` | `f8a1cb5ad6c9bd8df5b68a6016405a98168e7978` | MIT license visible in repository. Optional cross-engine rules oracle only. |

## Immediate source conclusions

- Use the 2009 Constant table as the paper-backed P0 target, but independently implement its complete scoring and tie behavior.
- Do not implement the guide’s 2010 Linear values as “the paper champion” without resolving the contradiction. The verified 2010 Table 6 values should be retained as the default paper record.
- The Sketch repository is sufficiently available for a later source inspection, but no port should begin until its actual DSL, AST, saved programs, demonstrations, and attribution requirements are recorded.
- The Sketch source inspection found a Node-based AST with arithmetic, conditionals, map/sum/argmax, state primitives, and separate pair/stop program roots. The repository contains demonstrations (`human_matches.pkl`, `glenn_matches.pkl`, `random_matches.pkl`) but no serialized champion program in the audited tree. This project adds a deliberately smaller JSON AST interpreter and classifies it as an adaptation; it does not load the repository's pickle data or claim to reproduce a synthesized policy.
- `vmerckle_monte` and `vmerckle_monte_uct` are clean-room reimplementations. `monte.cpp` plus `ai.cpp`/`cantstop.cpp` establish flat action sampling, random full-game rollouts, and root UCB1; the latter is intentionally not called full UCT. The audited commit has no explicit license, so no source was copied.
- Carter is architecture/checkpoint verification work, not a Phase 1 prerequisite.
- Carter verification now supports an optional inference-only Keras adapter. Its feature mapping uses this engine's equivalent rules and is explicitly an adaptation; no Carter checkpoint was downloaded into this project and no Carter result is claimed.
