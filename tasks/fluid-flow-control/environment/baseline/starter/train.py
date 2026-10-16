"""Train/resume the same shared PPO across successive native CFD case phases."""

import json
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from publish import publish
from shared import SharedEnv, dimensions


def train(env, suite, case, steps, resume, submission, publish_every=2000):
    config = dimensions(suite)
    resume = Path(resume)
    resume.parent.mkdir(parents=True, exist_ok=True)
    config_path = resume.with_suffix(".json")
    if resume.exists() and (not config_path.is_file() or json.loads(config_path.read_text()) != config):
        raise ValueError("Resume checkpoint has incompatible/missing shared-suite configuration")
    wrapped = SharedEnv(env, case, config)
    if resume.exists():
        model = PPO.load(resume, env=wrapped)
    else:
        model = PPO("MlpPolicy", wrapped, policy_kwargs={"net_arch": config["hidden"]},
                    n_steps=200, batch_size=40, learning_rate=3e-4, verbose=1)
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    def save():
        pending = resume.with_name(resume.stem + ".pending.zip")
        model.save(pending)
        os.replace(pending, resume)
        publish(model.policy, config, submission, model.num_timesteps)

    class PublishUpdates(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_start(self):
            # Rollout-start follows the preceding PPO update, unlike rollout-end.
            if self.model.num_timesteps and self.model.num_timesteps % publish_every == 0:
                save()

    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=PublishUpdates())
        save()
    finally:
        wrapped.close()
