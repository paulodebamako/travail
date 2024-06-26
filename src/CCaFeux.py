import traci
from Feu import Feu
from numpy import mean

class CCaFeux():
    def __init__(self, id, env) -> None:
        self.id = id
        self.env = env
        self.dict_feux = self._create_dict_feux()

    def _create_dict_feux(self) -> dict:
        """Create a dictionary of the form {num_feu : Feu}

        Returns:
            dict: _description_
        """
        self.sep = '__'
        dict_feux = {}
        for idx, list_link in enumerate(traci.trafficlight.getControlledLinks(self.get_id())):
            id_feu = self.id +self.sep+str(idx)
            if len(list_link) >= 1:
                dict_feux[id_feu] = Feu(env=self.env, id=id_feu, cc_a_feux=self)
        return dict_feux
    
    def get_id(self):
        """Return the id of the CCaFeux"""
        return self.id
    
    def get_number_of_feux(self) -> int:
        """Return the number of feux"""
        return len(self.dict_feux)
    
    def get_sum_waiting_time(self) -> float:
        """Return the sum of waiting time (in sec) of all vehicles at the intersection (need to check if it is summed or meaned in sumo)"""
        waiting_time = 0
        for feu in self.dict_feux.values():
            waiting_time += feu._get_lane_waiting_time()
        return waiting_time

    def get_mean_waiting_time(self) -> float:
        """Return the mean of waiting time (in sec) of all vehicles at the intersection (need to check if it is summed or meaned in sumo)"""
        waiting_time = []
        #nb_veh_emergency_stop = traci.simulation.getEmergencyStoppingVehiclesNumber()
        # if nb_veh_emergency_stop > 0:
        #     penalty = 1_000_000
        # else:
        #     penalty = 1
        for feu in self.dict_feux.values():
            waiting_time.append(feu._get_lane_waiting_time())
        return mean(waiting_time) #* -1 * penalty