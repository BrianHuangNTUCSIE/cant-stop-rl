# Documentation consistency audit

Reviewed the root README, simulator README, and `docs/EXPERIMENT_STATUS.md`
after the final evidence package was created.

## Resolved

- The root `README.md` is the project narrative and headline comparison.
- `cant_stop_baselines/README.md` is the implementation/API guide and no
  longer contains an incomplete performance matrix.
- PPO-451 + GA, DDQN, and Ridge are the retained final model set. Random
  Forest and failed PPO branches are not presented as promoted models.
- Learned pairing is described as a negative result, with the common-seed
  comparison and paired intervals recorded in the status document.
- vmerckle results are labeled as descriptive and root-only UCB1; they are not
  presented as full-tree MCTS.
- Formal held-out results, descriptive budget sweeps, and negative diagnostics
  are separated in `final_evidence.csv` and `FINAL_EVIDENCE.md`.

## Interpretation boundaries retained

- The post-merge README table uses PPO-460/DDQN-448/Ridge-642 and is a direct
  comparison against prior agents.
- The final evidence package uses the common held-out evaluation of PPO-451,
  DDQN, and Ridge for the primary model claim. These are different runs and
  are not silently merged.
- Paired intervals refer to shared dice-seed seat-swapped units, not to all
  individual games as independent observations.
- Two additional training seeds support a robustness check, but do not justify
  claiming universal dominance across all possible training seeds.

No unresolved numerical contradiction was found in the selected final claims.
