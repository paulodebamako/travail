from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
#from FpfEnv import FeuParFeuEnv
from CustomCallbacks import CustomCallbacks
from FpfEnv_v2 import FeuParFeuEnv
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray import tune
import os


ray.init()

net_file = r"D:\mesdocuments\paul\gertrude\travail\R&D\projets\generation_diag\travail\data\networks\2way-single-intersection\single-intersection.net.xml"
route_file = r"D:\mesdocuments\paul\gertrude\travail\R&D\projets\generation_diag\travail\data\networks\2way-single-intersection\single-intersection.rou.xml"

register_env('fpf', lambda config : ParallelPettingZooEnv((FeuParFeuEnv(
        net_file=net_file,
        route_file=route_file,
        reward_fn_name='arrived_vehicles',
        with_gui=True,
        num_seconds=3600
))))


num_rollout_workers = 4
rollout_fragment_length = 128
train_batch_size = num_rollout_workers * rollout_fragment_length
config = (
        PPOConfig()
        .environment(env='fpf', clip_actions=True)
        .rollouts(num_rollout_workers=num_rollout_workers, rollout_fragment_length=rollout_fragment_length)
        .training(
            train_batch_size=train_batch_size,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

tune.run(
        "PPO",
        name="PPO",
        stop={"num_episodes": 5 if not os.environ.get("CI") else 2},
        checkpoint_freq=50,
        local_dir=r"D:\mesdocuments\paul\gertrude\travail\R&D\projets\generation_diag\travail\outputs\fpf",
        config=config.to_dict(),
    )