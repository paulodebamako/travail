from FpfEnv_v2 import FeuParFeuEnv
from ray.rllib.examples.envs.classes.action_mask_env import ActionMaskEnv
import supersuit as ss
from stable_baselines3 import PPO
import os
#from pettingzoo.butterfly.knights_archers_zombies_v10
from stable_baselines3.common.vec_env import VecMonitor
from supersuit.vector import MarkovVectorEnv
from ray.rllib.examples.envs.classes.action_mask_env import ActionMaskEnv
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from pettingzoo.test import parallel_api_test, api_test,  parallel_seed_test, seed_test
from pettingzoo.utils.conversions import parallel_to_aec


def env_creation():
    net_file = r".\data\networks\2way-single-intersection\single-intersection.net.xml"
    route_file = r".\data\networks\2way-single-intersection\single-intersection.rou.xml"

    env = FeuParFeuEnv(
        net_file=net_file,
        route_file=route_file,
        reward_fn_name='arrived_vehicles',
        with_gui=True,
        sumo_seed=None,
        num_seconds=3600
    )
    return env

def launch_aec_api_test():
    par_env = env_creation()
    aec_env = parallel_to_aec(par_env=par_env)
    api_test(env=aec_env)


def launch_parallel_api_test():
    parallel_api_test(par_env=env_creation(), num_cycles=1000)

launch_aec_api_test()



