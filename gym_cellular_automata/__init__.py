from gym.envs.registration import register

from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

REGISTERED_CA_ENVS = (
    "ForestFireHelicopter-v0",
    "ForestFireBulldozer-v1",
)

ff_dir = "gym_cellular_automata.forest_fire"

register(
    id=REGISTERED_CA_ENVS[0],
    entry_point=ff_dir + ".helicopter:ForestFireEnvHelicopterV0",
)


register(
    id=REGISTERED_CA_ENVS[1],
    entry_point=ff_dir + ".bulldozer:ForestFireEnvBulldozerV1",
)

__all__ = ["CAEnv", "Operator", "GridSpace", "REGISTERED_CA_ENVS"]
