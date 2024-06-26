from FpfEnv_v2 import FeuParFeuEnv
import cloudpickle
import copy
import sumo_rl

def env_fpf_creation():
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

def env_sumo_rl_creation():
    RESOLUTION = (3200, 1800)
    env = sumo_rl.grid4x4(use_gui=True, out_csv_name="outputs/grid4x4/ppo_test", virtual_display=RESOLUTION)
    
    return env

# env_fpf = env_fpf_creation()
# env_fpf_copy = cloudpickle.loads(cloudpickle.dumps(env_fpf))
# env_sumo_rl = env_sumo_rl_creation()
# env_sumo_rl_copy = cloudpickle.loads(cloudpickle.dumps(env_sumo_rl))

# print('******** ENV FPF *********')
# print(f"env_fpf.label : {env_fpf.label}")
# print(f"env_fpf_copy.label : {env_fpf_copy.label}")
# print('******** ENV SUMO RL *********')
# print(f"env_sumo_rl.label : {env_sumo_rl.label}")
# print(f"env_sumo_rl_copy.label : {env_sumo_rl_copy.label}")


def test_label_diff():
    net_file = r".\data\networks\2way-single-intersection\single-intersection.net.xml"
    route_file = r".\data\networks\2way-single-intersection\single-intersection.rou.xml"

    env1 = env_fpf_creation()


    env2 = cloudpickle.loads(cloudpickle.dumps(env1))

    print(f"env1.label : {env1.label}")
    print(f"env2.label : {env2.label}")


class Test():
    label = 0

    def __init__(self) -> None:
        self.label = Test.label
        Test.label += 1

def t():
    t1 = Test()
    #t2 = cloudpickle.loads(cloudpickle.dumps(t1))
    t2 = copy.deepcopy(t1)
    print(f"t2 : {t2}")
    print(f"t1.label : {t1.label}")
    print(f"t2.label : {t2.label}")

if __name__ == '__main__':
    t()