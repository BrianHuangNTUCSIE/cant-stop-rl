# Can't Stop Browser Demo

This is a dependency-free frontend demo. You can select agents for both players, including the project's DDQN, Ridge, PPO, Rule of 28, Glenn & Aloi, Glenn 2009/2010, Couto, Random, and browser adaptations of Monte Carlo and root-UCB search. Open `index.html` in a browser, or run:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. An HTTP server is required because browsers usually block ES modules and JSON loaded from `file://` URLs.

## Published trained models

The demo publishes browser-loadable JSON weights instead of binary `.pt` files. `models/dqn.json` contains DDQN weights, `models/ppo.json` contains the domain-feature PPO export, and `models/ridge.json` contains the domain-feature Ridge scaler and coefficients. Sources, seeds, feature versions, and checksums are documented in [`models/README.md`](models/README.md).

To re-export a model in an environment with PyTorch installed, run:

```bash
../.venv/bin/python tools/export_weights.py \
  ../cant_stop_baselines/runs/ddqn_mc_counterfactual_seed447/best.pt \
  models/dqn.json
```

If you run this from the repository's `final` directory, adjust the checkpoint path as needed. After the JSON is generated, the page loads the model automatically. If a model is missing, the page reports that clearly and uses a heuristic fallback. The published PPO and Ridge files are included in this repository.

## GitHub Pages

When a push to `main` changes `web_demo/`, `.github/workflows/deploy-pages.yml` publishes `web_demo/` directly to GitHub Pages. The workflow can also be started manually. In the repository, set **Settings → Pages → Build and deployment → Source** to **GitHub Actions**. Deployment does not require npm, PyTorch, or a Python server.

Live demo: <https://brianhuangntucsie.github.io/cant-stop-rl/>. If Pages is not enabled, complete the repository setup above first.

The browser demo is for interactive demonstration only. Its JavaScript randomness and smaller search-rollout budgets are not equivalent to the formal Python evaluation and must not replace the results in the report.

### Browser performance benchmark

For development-only browser timing, serve the repository and open
`/tools/benchmark_vmerckle.html`. Click **Run benchmark** to measure rollout
throughput and generation cancellation latency in the current browser. This
page is not part of the application UI and does not change the production
search budget or policy.

### Agent fidelity

The game rules, legal-move ordering, Rule of 28, Glenn & Aloi, Glenn 2009 Constant, Glenn 2010 Linear, and Couto pairing/Stop decisions match the current Python engine. Tie handling for paper agents also preserves Python's first-max behavior.

Random uses a random policy, as in Python. `mc`, `mc-full-game`, `mc-strong`, `vmerckle_monte`, and `vmerckle_monte_uct` each have a corresponding root/action search. The browser versions use an unseeded JavaScript RNG and smaller rollout budgets to remain interactive; formal report results must still use the original Python simulator.

The browser `snapshot()` returns copied data, so agents and debug code cannot mutate the live game through a snapshot. The PPO exporter exports the policy head rather than the value head, allowing the generated `models/ppo.json` to run directly in the browser.

The **Quick test** panel runs a user-selected number of headless AI-vs-AI games without rendering each move, then reports both players' win counts and win rates. Select AI agents for both players before starting a test.

An eight-action PPO checkpoint uses logits 0–5 to choose a sorted legal pairing during the pair phase and logits 6–7 during the decision phase. A two-action PPO checkpoint continues to use GA pairing.

The interface lets Player A and Player B select agents independently. Selecting **Human** enables a local two-player game; selecting two AI agents plays the entire game automatically.

## Architecture

- `js/game.js`: Browser game state machine and GA dice pairing
- `js/model.js`: Dependency-free MLP and Ridge inference
- `js/app.js`: UI and AI turn lifecycle

The Ridge policy uses the separate `feature_count`/`feature_mode` metadata
schema and the 68-value domain encoder; neural policies use their
`observation_size`/`action_size` schema. Invalid Ridge feature metadata or
vector lengths are rejected rather than silently substituting another policy.
- `tools/export_weights.py`: PyTorch checkpoint to browser JSON conversion

Model weights are public because this is a frontend-only application. This design is not suitable for games that need to protect models or prevent cheating.

The Traditional Chinese source is preserved in [`README.zh-TW.md`](README.zh-TW.md).
