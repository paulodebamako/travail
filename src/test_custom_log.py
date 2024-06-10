# TODO (sven): Move this example script into the new API stack.

"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.

We then use `keep_per_episode_custom_metrics` to keep the per-episode values
of our custom metrics and do our own summarization of them.
"""

from typing import Dict, Tuple
import argparse
import gymnasium as gym
import numpy as np
import os

import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from FpfEnv_v2 import FeuParFeuEnv
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument("--stop-iters", type=int, default=2000)


# Create a custom CartPole environment that maintains an estimate of velocity
# class CustomCartPole(gym.Env):
#     def __init__(self, config):
#         self.env = gym.make("CartPole-v1")
#         self.observation_space = self.env.observation_space
#         self.action_space = self.env.action_space
#         self._pole_angle_vel = 0.0
#         self.last_angle = 0.0

#     def reset(self, *, seed=None, options=None):
#         self._pole_angle_vel = 0.0
#         obs, info = self.env.reset()
#         self.last_angle = obs[2]
#         return obs, info

#     def step(self, action):
#         obs, rew, term, trunc, info = self.env.step(action)
#         angle = obs[2]
#         self._pole_angle_vel = (
#             0.5 * (angle - self.last_angle) + 0.5 * self._pole_angle_vel
#         )
#         info["pole_angle_vel"] = self._pole_angle_vel
#         return obs, rew, term, trunc, info


net_file = r"D:\mesdocuments\paul\gertrude\travail\R&D\projets\generation_diag\travail\data\networks\2way-single-intersection\single-intersection.net.xml"
route_file = r"D:\mesdocuments\paul\gertrude\travail\R&D\projets\generation_diag\travail\data\networks\2way-single-intersection\single-intersection.rou.xml"

register_env('fpf', lambda config : ParallelPettingZooEnv((FeuParFeuEnv(
        net_file=net_file,
        route_file=route_file,
        reward_fn_name='arrived_vehicles',
        with_gui=True,
        num_seconds=3600
))))

class CustomCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        print(f"episode.length : {episode.length}")
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )

        # list of custom metrics
        custom_metrics = ['mean_waiting_time_cc']
        # Create lists to store custom metrics
        for metric in custom_metrics:
            episode.user_data[metric] = []
            episode.hist_data[metric] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        
        for metric in base_env.custom_metrics:
            episode.user_data[metric].append(episode.last_info_for()[metric])

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.config.batch_mode == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        for metric in base_env.custom_metrics:
            episode.custom_metrics[metric] = np.mean(episode.user_data[metric])
            episode.hist_data[metric] = episode.user_data[metric]

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.config.batch_mode == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        pole_angle = np.mean(episode.user_data["pole_angles"])
        episode.custom_metrics["pole_angle"] = pole_angle
        episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        # We can also do our own sanity checks here.
        assert (
            samples.count == 2000
        ), f"I was expecting 2000 here, but got {samples.count}!"

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

        # Normally, RLlib would aggregate any custom metric into a mean, max and min
        # of the given metric.
        # For the sake of this example, we will instead compute the variance and mean
        # of the pole angle over the evaluation episodes.
        custom_metrics = result[ENV_RUNNER_RESULTS]["custom_metrics"]
        pole_angle = custom_metrics["pole_angle"]
        var = np.var(pole_angle)
        mean = np.mean(pole_angle)
        custom_metrics["pole_angle_var"] = var
        custom_metrics["pole_angle_mean"] = mean
        # We are not interested in these original values
        del custom_metrics["pole_angle"]
        del custom_metrics["num_batches"]

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        result["sum_actions_in_train_batch"] = train_batch["actions"].sum()
        # Log the sum of actions in the train batch.
        print(
            "policy.learn_on_batch() result: {} -> sum actions: {}".format(
                policy, result["sum_actions_in_train_batch"]
            )
        )

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs,
    ):
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1


if __name__ == "__main__":
    args = parser.parse_args()

    net_file = r"D:\mesdocuments\paul\gertrude\travail\R&D\projets\generation_diag\travail\data\networks\2way-single-intersection\single-intersection.net.xml"
    route_file = r"D:\mesdocuments\paul\gertrude\travail\R&D\projets\generation_diag\travail\data\networks\2way-single-intersection\single-intersection.rou.xml"

    register_env('fpf', lambda config : ParallelPettingZooEnv((FeuParFeuEnv(
            net_file=net_file,
            route_file=route_file,
            reward_fn_name='arrived_vehicles',
            with_gui=True,
            num_seconds=3600
    ))))

    config = (
        PPOConfig()
        .environment(env='fpf')
        .framework(args.framework)
        .callbacks(CustomCallbacks)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .env_runners(enable_connectors=False)
        .reporting(keep_per_episode_custom_metrics=True)
    )

    ray.init(local_mode=True)
    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={TRAINING_ITERATION: args.stop_iters},
        ),
        param_space=config,
    )
    # there is only one trial involved.
    result = tuner.fit().get_best_result()

    # Verify episode-related custom metrics are there.
    custom_metrics = result.metrics["env_runners"]["custom_metrics"]
    print(custom_metrics)
    assert "pole_angle_mean" in custom_metrics
    assert "pole_angle_var" in custom_metrics 