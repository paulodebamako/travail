from FpfEnv_v2 import FeuParFeuEnv
from stable_baselines3 import PPO
import supersuit as ss
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

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

env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

model.learn(total_timesteps=5)

env.close()