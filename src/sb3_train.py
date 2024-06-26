from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from FpfEnv_v2 import FeuParFeuEnv
import supersuit as ss

import sumo_rl

def main():

    def env_creation():
        net_file = r".\data\networks\2way-single-intersection\single-intersection.net.xml"
        route_file = r".\data\networks\2way-single-intersection\single-intersection.rou.xml"

        env = FeuParFeuEnv(
            net_file=net_file,
            route_file=route_file,
            reward_fn_name='mean_waiting_time',
            with_gui=True,
            sumo_seed=None,
            num_seconds=3600
        )
        return env

    init_env = env_creation()

    env = ss.pettingzoo_env_to_vec_env_v1(init_env)
    env = ss.concat_vec_envs_v1(vec_env=env, num_vec_envs=2, num_cpus=1, base_class="stable_baselines3")

    model = MaskablePPO("MultiInputPolicy", env=env)
    print("/////// MODEL LEARN ///////")
    model.learn(3600)

    # model.save(r"D:\mesdocuments\paul\gertrude\travail\R&D\projets\generation_diag\travail\outputs\fpf")


if __name__ == "__main__":
    main()



