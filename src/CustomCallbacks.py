from typing import Dict
import numpy as np
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy import Policy
import unittest
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import framework_iterator

class CustomCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        #print(f"episode.length : {episode.length}")
        # assert episode.length == 0, (
        #     "ERROR: `on_episode_start()` callback should be called right "
        #     "after env reset!"
        # )

        # list of custom metrics
        custom_metrics = ['mean_waiting_time_cc']
        # Create lists to store custom metrics
        for metric in custom_metrics:
            episode.user_data[metric] = []
            episode.hist_data[metric] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        # assert episode.length > 0, (
        #     "ERROR: `on_episode_step()` callback should not be called right "
        #     "after env reset!"
        # )
        
        for metric in base_env.custom_metrics:
            episode.user_data[metric].append(episode.last_info_for()[metric])

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        # if worker.config.batch_mode == "truncate_episodes":
        #     # Make sure this episode is really done.
        #     assert episode.batch_builder.policy_collectors["default_policy"].batches[
        #         -1
        #     ]["dones"][-1], (
        #         "ERROR: `on_episode_end()` should only be called "
        #         "after episode is done!"
        #     )
        for metric in base_env.custom_metrics:
            episode.custom_metrics[metric] = np.mean(episode.user_data[metric])
            episode.hist_data[metric] = episode.user_data[metric]


# class TestCallbacks(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         ray.init()

#     @classmethod
#     def tearDownClass(cls):
#         ray.shutdown()

#     def test_episode_and_sample_callbacks(self):
#         config = (
#             PPOConfig()
#             .environment("CartPole-v1")
#             .env_runners(num_env_runners=0)
#             .callbacks(CustomCallbacks)
#             .training(train_batch_size=50, sgd_minibatch_size=50, num_sgd_iter=1)
#         )
#         for _ in framework_iterator(config, frameworks=("torch")):
#             algo = config.build()
#             algo.train()
#             algo.train()
#             callback_obj = algo.workers.local_worker().callbacks
#             self.assertGreater(callback_obj.counts["sample"], 0)
#             self.assertGreater(callback_obj.counts["start"], 0)
#             self.assertGreater(callback_obj.counts["end"], 0)
#             self.assertGreater(callback_obj.counts["step"], 0)
#             algo.stop()


# TestCallbacks().test_episode_and_sample_callbacks()