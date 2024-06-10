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
    env = parallel_to_aec(env)
    return env


net_file = r".\data\networks\2way-single-intersection\single-intersection.net.xml"
route_file = r".\data\networks\2way-single-intersection\single-intersection.rou.xml"

# par_env = FeuParFeuEnv(
#     net_file=net_file,
#     route_file=route_file,
#     reward_fn_name='arrived_vehicles',
#     with_gui=True,
#     sumo_seed=None,
#     num_seconds=3600
# )

api_test(env=env_creation())

#parallel_api_test(par_env=par_env)


# par_env.reset()
# for _ in range(par_env.num_seconds):
#     actions = {agent: par_env.action_space(agent).sample() for agent in par_env.agents}
#     observations, rewards, terminations, truncations, infos = par_env.step(actions)

# par_env.close()



# print(f"par_env.label : {par_env.label}")
# print(f'par_env.CONNECTION_LABEL : {par_env.CONNECTION_LABEL}')
#aec_env = pettingzoo.utils.conversions.parallel_to_aec(par_env = par_env)

# env = ss.pettingzoo_env_to_vec_env_v1(par_env)
#env = MarkovVectorEnv(par_env)
#print(f"env.num_envs : {env.num_envs}")
# print(type(env))
#env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class='stable_baselines3')
#print(type(env))

#env = VecMonitor(env)
# print('etape 5')
# model = PPO(
#         "MlpPolicy",
#         env,
#         verbose=2,
#         gamma=0.95,
#         n_steps=256,
#         ent_coef=0.0905168,
#         learning_rate=0.00062211,
#         vf_coef=0.042202,
#         max_grad_norm=0.9,
#         gae_lambda=0.99,
#         n_epochs=2,
#         clip_range=0.3,
#         batch_size=256,
#         tensorboard_log="./logs/grid4x4/ppo_test",
#     )

# print("Starting training")
# model.learn(total_timesteps=10)
# print('etape 6')
# model.learn(total_timesteps=par_env.sim_max_time)
# print('etape 7')
# model.save(os.path.join('saved_models', 'test_intersection_simple'))

