# Published browser models

These are the published inference exports used by the browser demo. They contain
only the policy networks needed for browser-side inference, not PyTorch
checkpoints or optimizer state.

| File | Source checkpoint | Seed | Features | SHA-256 |
|---|---|---:|---|---|
| `dqn.json` | `cant_stop_baselines/runs/ddqn_mc_counterfactual_seed447/best.pt` | 447 | domain (68 observation values) | `639bb445f5b26361f3c22461803897b7196ffbbb56f1b610e752419ed964e240` |
| `ppo.json` | `cant_stop_baselines/runs/ppo_mc_focused_seed460/best.pt` | 460 | domain (68 observation values) | `58c2fc11f8a194505a868419f174e9967bffe5fac3edf6866d398e6e6cb1bb65` |

The checkpoint path and seed are provenance metadata, not a claim that the
browser's lightweight runtime reproduces the Python evaluation exactly.
Both files are loaded automatically by the UI. If either file is unavailable
or invalid, its corresponding agent is disabled and the demo reports that
model as unavailable. To verify local copies, run:

```bash
shasum -a 256 web_demo/models/dqn.json web_demo/models/ppo.json
```
