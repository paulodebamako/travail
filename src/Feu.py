import traci
import traci.domain
from gymnasium.spaces import Discrete
import numpy as np
import sumo_rl

class Feu():
    def __init__(self, env, id: int, cc_a_feux) -> None:
        self.env = env
        self.id = id
        self.cc_a_feux = cc_a_feux
        self.observation_fn = self.env.observation_class(self)
        self.observation_space = self.observation_fn.observation_space()
        self.reward_fn_name = self.env.reward_fn_name
        self._check_reward()
        self._check_custom_metrics()
        self.possible_actions = ['g', 'y', 'r']
        self.consecutive_durations = {state : 0 for state in self.possible_actions}
        self.cumulated_durations = {state : 0 for state in self.possible_actions}
        
    def _check_custom_metrics(self):
        """Check if the custom metrics passed in the params Env existed"""
        for metric in self.env.custom_metrics:
            if metric not in self.get_possible_reward_dict().keys():
                raise Exception(f"Metric {metric} does not exist")
            
    def get_action_space(self):
        """Return the action space"""
        return Discrete(3)
    
    def update(self, state: str):
        """"Update all the values related to this Feu such as cumulated green time etc
        Args:
            state (str): should be 'g', 'y' or 'r' 
        """
        new = {key : value + 1 if key == state else 0 for key, value in self.consecutive_durations.items()}
        self.consecutive_durations.update(new)
        self.cumulated_durations[state] += 1

    def set_action(self, action):
        """Set the new state in SUMO"""
        traci.trafficlight.setLinkState(
            tlsID=self.get_cc_a_feux_id(),
            tlsLinkIndex=self.get_numero(),
            state=self.possible_actions[action]
            )
        self.update(self.possible_actions[action])

    def _check_reward(self):
        """Raise error if reward is unfeasible"""
        if self.reward_fn_name not in self.get_possible_reward_dict().keys():
            raise Exception(f'Error, the reward name {self.reward_fn_name} does not exist')

    def get_numero(self) -> int:
        """Return the numero of the feu"""
        return int(self.id.split(self.cc_a_feux.sep)[1])

    def get_cc_a_feux_id(self) -> str:
        """Return the id of the cc_a_feux the feu belongs to"""
        return self.id.split(self.cc_a_feux.sep)[0]

    def get_current_couleur(self) -> str:
        """Return the actual state (SUMO pov) of the Feu. Must be in self.possible_actions """
        return traci.trafficlight.getRedYellowGreenState(tlsID=self.get_cc_a_feux_id())[self.get_numero()]
        
    def _get_edge(self):
        """Returns the edge id the feu belongs to"""
        pass
    
    def get_lane_id(self) -> str:
        """Returns the lane id the feu belongs to"""
        return traci.trafficlight.getControlledLinks(self.cc_a_feux.get_id())[self.get_numero()][0][0]

    def get_last_step_halting_number(self) -> int:
        """Returns the total number of halting vehicles for the last time step 
        """
        return traci.lane.getLastStepHaltingNumber(laneID=self.get_lane_id())
    
    def get_last_step_mean_speed(self) -> int:
        return traci.lane.getLastStepMeanSpeed(laneID=self.get_lane_id())

    def compute_observation(self):
        """Compute the observation for the Feu"""
        return self.observation_fn()
    
    def compute_reward(self) -> float:
        """Compute and return the reward the Feu"""
        reward = self.get_possible_reward_dict()[self.reward_fn_name]()
        return reward

    def compute_infos(self) -> dict:
        """Compute the infos for the Feu"""
        infos = {metric : self.get_possible_reward_dict()[metric] for metric in self.env.custom_metrics}
        #infos = {self.id : {}}
        return infos
    
    def compute_action_mask(self) -> np.array:
        """Return the action mask for the Feu.
        Should be a list or array with
        1 : valid action
        0 : invalid action
        """
        action_mask = None
        current_state = self.get_current_couleur()
        if current_state == "g":
            # ctr vert min
            if self.consecutive_durations["g"] < self.env.min_green_time:
                action_mask = np.array([1, 0, 0], dtype=np.int8)
            else:
                action_mask = np.array([1, 1, 0], dtype=np.int8)
        elif current_state == "y":
            # ctr orange 
            if self.consecutive_durations["y"] < self.env.yellow_time:
                action_mask = np.array([0, 1, 0], dtype=np.int8)
            elif self.consecutive_durations["y"] == self.env.yellow_time:
                action_mask = np.array([0, 0, 1], dtype=np.int8)
            else:
                raise Exception(f"Impossible d'avoir plus de 3 sec de orange.\nself.consecutive_durations : {self.consecutive_durations['y']}")
        else:
            # ctr rouge max
            if self.consecutive_durations["r"] >= self.env.max_red_time:
                action_mask = np.array([1, 0, 0], dtype=np.int8)
            else:
                action_mask = np.array([1, 0, 1], dtype=np.int8)
        return action_mask
    
    def _get_lane_waiting_time(self):
        """Returns the waiting time for all vehicles on the lane that the Feu belongs to"""
        return traci.lane.getWaitingTime(self.get_lane_id())
    
    def _green_reward(self) -> float:
        """ Return +1, 0, -1 if the last action was green, yellow, red"""
        if self.env.last_actions[self.id] == 'g':
            reward = 1
        elif self.env.last_actions[self.id] == 'y':
            reward = 0
        else:
            reward = -1
        return reward
    
    def _arrived_vehicles_reward(self) -> float:
        """Return the current number of vehicles arrived at their destinations"""
        return traci.simulation.getArrivedNumber()
    
    def get_possible_reward_dict(self):
        """Return the dict of the form {reward_fn_name : reward_fn}"""
        reward_fn = {
            "green" : self._green_reward,
            "arrived_vehicles" : self._arrived_vehicles_reward,
            "mean_waiting_time_cc" : self.cc_a_feux.get_mean_waiting_time
        }
        return reward_fn
