# Import necessary modules
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
from ray.rllib.env import ParallelPettingZooEnv
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray import tune
from ray.tune.registry import register_env

class PrisonersDilemmaParallel(ParallelEnv):
    def __init__(self):
        self.possible_agents = ["agent_0", "agent_1"]
        self.action_spaces = {agent: spaces.Discrete(2) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(2,), dtype=np.int8) for agent in self.possible_agents
        }
        self.reset()

    def reset(self):
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.cumulative_reward = {agent : 0 for agent in self.possible_agents}
        self.observations = {agent: np.array([0, 0]) for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        return self.observations

    def step(self, actions):
        if not all(agent in actions for agent in self.possible_agents):
            raise ValueError("All agents must have an action")

        action_0, action_1 = actions["agent_0"], actions["agent_1"]

        if action_0 == 0 and action_1 == 0:
            rewards = [1, 1]
        elif action_0 == 1 and action_1 == 0:
            rewards = [10, 0]
        elif action_0 == 0 and action_1 == 1:
            rewards = [0, 10]
        else:
            rewards = [0, 0]

        self.rewards = {"agent_0": rewards[0], "agent_1": rewards[1]}
        for agent, cumul_reward in self.rewards.items():
            self.cumulative_reward[agent] += self.rewards[agent]
        self.dones = {"agent_0": False, "agent_1": False}
        self.observations = {
            "agent_0": np.array([actions["agent_0"], actions["agent_1"]]),
            "agent_1": np.array([actions["agent_1"], actions["agent_0"]]),
        }
        return self.observations, self.rewards, self.dones, self.infos

    def render(self):
        print(f"Agent 0: {self.observations['agent_0']}, Turn Reward: {self.rewards['agent_0']}, Cumul Reward : {self.cumulative_reward['agent_0']}")
        print(f"Agent 1: {self.observations['agent_1']}, Turn Reward: {self.rewards['agent_1']}, Cumul Reward : {self.cumulative_reward['agent_1']}")

    def close(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
def launch_qq_tours(nb_tours: int = 5):
    # Initialize the environment
    env = PrisonersDilemmaParallel()

    # Run 5 turns of the game
    env.reset()
    for turn in range(5):
        actions = {
            "agent_0": np.random.choice([0, 1]),  # Random action for agent 0
            "agent_1": np.random.choice([0, 1])   # Random action for agent 1
        }
        observations, rewards, dones, infos = env.step(actions)
        print(f"Turn {turn + 1}:")
        env.render()


# Création de l'environnement
def env_creator(_):
    return PrisonersDilemmaParallel()

# Enregistrement de l'environnement
register_env("prisoners_dilemma_parallel", lambda config: env_creator(config))

num_rollout_workers = 4
rollout_fragment_length = 360
train_batch_size = num_rollout_workers * rollout_fragment_length
config = (
        PPOConfig()
        .environment(env='prisoners_dilemma_parallel', clip_actions=True)
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
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=0)
    )

# Initialisation de Ray
ray.init(ignore_reinit_error=True)

# Lancement de l'entraînement
tune.run(
    "PPO",
    stop={"episodes_total": 5000},
    config=config
)

# Arrêt de Ray
ray.shutdown()