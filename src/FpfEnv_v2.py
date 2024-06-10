import traci
import sumolib
import os
import sys
from random import randint
from pettingzoo import ParallelEnv
import functools
from typing import Union
from observations import ObservationFunction, DefaultObservationFunction
from CCaFeux import CCaFeux

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

class FeuParFeuEnv(ParallelEnv):
    metadata = {
        "name": "environnementfpf",
    }
    CONNECTION_LABEL = 0

    def __init__(
        self, 
        net_file: str, 
        route_file: str, 
        reward_fn_name: str,
        observation_class: ObservationFunction = DefaultObservationFunction,
        with_gui: bool = False,
        begin_time: int = 0,
        num_seconds: int = 3600,
        min_green_time: int = 6,
        yellow_time: int = 3,
        max_red_time: int = 120,
        sumo_seed: Union[None, str] = None,
        render_mode: str = "human",
        custom_metrics = ['mean_waiting_time_cc']
    ) -> None:
        """Initialize the environment"""
        self._net_file = net_file
        self._route_file = route_file
        self.observation_class = observation_class
        self.reward_fn_name = reward_fn_name
        self.with_gui = with_gui
        self.begin_time = begin_time
        self.num_seconds = num_seconds
        self.sim_max_time = begin_time + num_seconds
        self.min_green_time = min_green_time
        self.yellow_time = yellow_time
        self.max_red_time = max_red_time
        self.render_mode = render_mode
        self.custom_metrics = custom_metrics
        self.sumo_seed = sumo_seed
        self.label = str(FeuParFeuEnv.CONNECTION_LABEL)
        FeuParFeuEnv.CONNECTION_LABEL += 1
        if self.with_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
        self.dict_cc_a_feux = self._create_dict_cc_a_feux()
        self.dict_feux = self._create_dict_feux()
        self.agents = self._create_agents()
        self.possible_agents = self._create_agents()
        self.last_actions = {a: None for a in self.dict_feux.keys()}
        self.sumo_connection = None
        self.episode = 0
        self.timestep = 0
        print(f"self.label_conn = {self.label}")

    def _seed(self, seed=None):
        """Define the seed of the environnement"""
        self.sumo_seed = randint(1, 1_000_000)
        
    def _start_simulation(self, seed=None):
        """Start the simulation """
        sumo_cmd = [self._sumo_binary, "-n", self._net_file, "-r", self._route_file]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == 'random':
            sumo_cmd.append("--random")
        elif seed is not None:
            self.sumo_seed = seed
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        else:
            self._seed()
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if self.with_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
        traci.start(sumo_cmd, label=self.label)
        self.sumo_connection = traci.getConnection(label=self.label)
        if self.with_gui or self.render_mode is not None:
            self.sumo_connection.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
        self.timestep = 0

    def reset(self, seed=None, options=None):
        """Reset the environment.
        Returns:
            observations, infos
        """
        # close the connection if needed
        self.close()
        self.episode += 1
        # initialize agents
        self.agents = self._create_agents()
        # start simulation
        self._start_simulation(seed=seed)
        # compute observations
        observations = self._compute_observations()
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = self._compute_infos()
        return observations, infos
        
    def close(self):
        """Close the simulation"""
        if self.episode != 0:
            traci.switch(label=self.label)
            traci.close()
            self.sumo_connection = None

    def _create_dict_cc_a_feux(self) -> dict:
        """Create the dictionary that contains CCaFeux objects

        Returns:
            dict: dict of the form {tls_id: CCaFeux}
        """
        label_conn = f'init_connection_{self.label}'
        traci.start([sumolib.checkBinary("sumo"), "-n", self._net_file], label=label_conn)
        for tls_id in list(traci.trafficlight.getIDList()):
            dict_cc_a_feux = {
                tls_id: CCaFeux(id=tls_id, env=self)
            }
        traci.getConnection(label=label_conn).close()
        return dict_cc_a_feux

    def _create_dict_feux(self) -> dict:
        """Return the dictionary of all Feu"""
        dict_feux = {}
        for cc_a_feux in self.dict_cc_a_feux.values():
            dict_feux.update(cc_a_feux.dict_feux)
        return dict_feux
    
    def _create_agents(self) -> list:
        """Create the list that contains Feu objets

        Returns:
            list: list of all agents id
        """
        agents = []
        for cc_a_feux in self.dict_cc_a_feux.values():
            for id_feu in cc_a_feux.dict_feux.keys():
                agents.append(id_feu)
        return agents
    
    def _compute_observations(self) -> dict:
        """ Compute and return the observations
        Returns:
            observations (dict): dict of the form {id_agent: {'observation' : [], 'action_mask' : []}}
        """
        observations = {}
        for id_agent, agent in self.dict_feux.items():
            observations[id_agent] = {
                "observation" : agent.compute_observation(), 
                "action_mask" : agent.compute_action_mask()
            }
        return observations

    def _compute_rewards(self) -> dict:
        """Compute and return the reward
        Returns;
            rewards (dict): dict of the form {id_agent: reward}"""
        rewards = {}
        for id_agent, agent in self.dict_feux.items():
            rewards[id_agent] = agent.compute_reward()
        return rewards
    
    def _compute_terminations(self) -> dict:
        """Return the terminations dict"""
        terminations = {agent: False for agent in self.agents}
        return terminations
    
    def _compute_truncations(self) -> dict:
        """Return the truncations dict"""
        env_truncation = traci.simulation.getTime() >= self.sim_max_time
        truncations = {agent: env_truncation for agent in self.agents}
        return truncations
    
    def _compute_infos(self) -> dict:
        """Return the infos dict
        We stock all the different custom metrics in there. 
        So for each agent, it is a dictionary with the name of the custom_metrics as key and its value as value.
        """
        #infos = {id_agent: agent.compute_infos() for id_agent, agent in self.dict_feux.items()}
        infos = {id_agent : {} for id_agent in self.agents}
        return infos
    
    def _apply_actions(self, actions):
        """Set the given state for each Feu """
        self.last_actions = {id_feu : feu.possible_actions[actions[id_feu]] for id_feu, feu in self.dict_feux.items()}
        for id_feu, action in actions.items():
            self.dict_feux[id_feu].set_action(action)
    
    def step(self, actions):
        """Step (action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos, we will store custom metrics in there
        dicts where each dict looks like {agent_1: item_1}"""
        # apply actions
        self._apply_actions(actions)
        # compute rewards
        rewards = self._compute_rewards()
        # compute terminations
        terminations = self._compute_terminations()
        # compute truncations 
        truncations = self._compute_truncations()
        # compute observations
        observations = self._compute_observations()
        # compute infos
        infos = self._compute_infos()

        traci.simulationStep()
        self.timestep += 1

        # for feu, couleur_prec in self.actionprec.items():
        #     if couleur_prec == 'r':
        #         if self.red_duration[feu] >= 120:
        #             couleur = "G"
        #             self.red_duration[feu] = 0
        #         else:
        #             if actions[feu] == 0:
        #                  couleur = "r"
        #                  self.red_duration[feu] += 1
        #             else:
        #                 couleur = "G"
        #                 self.green_duration[feu] += 1
        #                 self.red_duration[feu] = 0
        #     elif couleur_prec == 'y':
        #         if self.orange_duration[feu] <= 3:
        #             couleur = 'y'
        #             self.orange_duration[feu] += 1
        #         else:
        #             couleur = "r"
        #             self.red_duration[feu] += 1
        #             self.orange_duration[feu] = 0
        #     else :
        #         if self.green_duration[feu] <= 6:

        #             couleur = 'G'
        #             self.green_duration[feu] += 1
        #         else:
        #             if actions[feu] == 0:
        #                  couleur = "G"
        #                  self.green_duration[feu] += 1
        #             else:
        #                 couleur = "y"
        #                 self.orange_duration[feu] += 1
        #                 self.green_duration[feu] = 0
        #     traci.trafficlight.setLinkState('t', int(feu[3:]), couleur)
        #     self.actionprec[feu] = couleur


        # Check termination conditions
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos


    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return the observation space of an agent ie a Feu
        """
        return self.dict_feux[agent].observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.dict_feux[agent].get_action_space()