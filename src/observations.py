"""Observation functions for a traffic signal link"""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from Feu import Feu


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, feu: Feu):
        """Initialize observation function."""
        self.feu = feu

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, feu: Feu):
        """Initialize default observation function."""
        super().__init__(feu)

    def __call__(self) -> np.ndarray:
        """Return the default observation.
        Return:
            np.array which represent a vector of shape (x,) ie 1 dimensional array
        """
        last_step_halting_number = [self.feu.get_last_step_halting_number()]
        last_step_mean_speed = [self.feu.get_last_step_mean_speed()]
        observation = np.array(last_step_halting_number + last_step_mean_speed, dtype=np.float32)
        #print(observation.shape)
        return observation

    def observation_space(self) -> spaces.Dict:
        """Return the observation space."""
        # return spaces.Box(
        #     low=0, 
        #     high=100,
        #     shape=(2,), 
        #     dtype=np.float32
        # )
        return spaces.Dict({
            "observation" : spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
            "action_mask" : spaces.MultiBinary(3)
        }
        )
        