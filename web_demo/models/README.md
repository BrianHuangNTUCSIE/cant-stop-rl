# Published browser models

`dqn.json` is the published DDQN inference export used by the browser demo.

| File | Source checkpoint | Seed | Features | SHA-256 |
|---|---|---:|---|---|
| `dqn.json` | `cant_stop_baselines/runs/ddqn_mc_counterfactual_seed447/best.pt` | 447 | domain (68 observation values) | `639bb445f5b26361f3c22461803897b7196ffbbb56f1b610e752419ed964e240` |

The checkpoint path and seed are provenance metadata, not a claim that the
browser's lightweight runtime reproduces the Python evaluation exactly.
The repository currently does **not** publish `ppo.json`; the UI disables the
PPO option until that file is present and successfully loaded. To verify a
local copy, run `shasum -a 256 web_demo/models/dqn.json` from the repository
root.
