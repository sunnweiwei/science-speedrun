# Agent-visible references

| Location | Purpose / validation |
| --- | --- |
| `/workspace/references/specialists` | Runnable, tested organizer PPO/DDPG/TD3 trainer, actor export and recipes on the fixed native profiles |
| `/workspace/references/results` | 39/39 completed specialist results: same physical metrics for uncontrolled, trained and paper references |
| `/workspace/baseline/starter` | Shared PPO train/resume/publication/inference; one parameter set across phases |
| `/opt/references/hydrogym-author` | Authors' public model/loss factories at `20a483ecc6ea711f6b380fff05f212ca4d7e2c67` |
| `/opt/hydrogym` | HydroGym physics source at `4ab9854dea3d84e38a59c25e0f5835a00cf8225f` |
| `/opt/references/tdmpc2` | Optional shared-world-model architecture reference at `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`; **not** a validated HydroGym baseline |

The image fetches pinned upstream source; attribution/licenses remain in the
source repositories. Large datasets and pretrained model parameters are not
included. Public-author API compatibility corrections and any recipe changes
are explicitly recorded by the specialist learner in each run.

Specialist controllers are comparison and allowed-data-generation resources.
Their result table is not one shared model's score, and bundling their weights
would violate the submission contract. Learn from their implementations and
training trajectories, then train one shared candidate on the permitted cases.

The task's current metrics are physical CD, L/D, load/angle error, TKE or wall
shear. References use the same row metric, not a training-reward proxy. Paper
curve readings are approximate and use different configurations/budgets; do
not treat them as exact seed-averaged leaderboard values or SOTA certification.
