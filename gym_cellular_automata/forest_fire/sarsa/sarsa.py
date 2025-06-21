from typing import Optional

import numpy as np
from gymnasium import logger, spaces

from gym_cellular_automata._config import TYPE_BOX
from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.forest_fire.operators import (
    ForestFire,
    Modify,
    Move,
    MoveModify,
)
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

#from .utils.render import render


class ForestFireSarsaEnv(CAEnv):
    metadata = {"render_modes": ["human"]}

    @property
    def SARS(self):
        return self._SARS

    @property
    def initial_state(self):
        if self._resample_initial:
            #self.grid = self.grid_space.sample()
            self.grid = self.grid_space.sample_det() # un inicio de fuego random
            #print(self.grid)
            ca_params = np.array([self._p_fire, self._p_tree], dtype=TYPE_BOX)
            #pos = np.array([self.nrows // 2, self.ncols // 2]) 
            pos = np.array([self.nrows - 1, (self.ncols// 3) - 2]) # la posición del agente
            pos_goal = np.array([1, self.ncols])
            freeze = np.array(self._max_freeze)
            self.context = ca_params, pos, pos_goal, freeze
            
            #print(f'pos {pos} pos goal {pos_goal}')

            self._initial_state = self.grid, self.context

        self._resample_initial = False

        return self._initial_state

    def __init__(
        self, nrows, ncols, speed: float = 0.5, freeze: Optional[int] = None, **kwargs
    ):
        # Sets defaults and runs seed method
        super().__init__(nrows, ncols, **kwargs)

        self.title = "ForestFireSars" + str(nrows) + "x" + str(ncols)

        # Env Representation Parameters

        actions = (
            up_left,
            up,
            up_right,
            left,
            not_move,
            right,
            down_left,
            down,
            down_right,
        ) = range(9)

        self._n_actions = len(actions)

        #self._reward_per_empty = kwargs.get("reward_per_empty",0.0)
        #self._reward_per_tree = kwargs.get("reward_per_tree", 1.0)
        #self._reward_per_fire = kwargs.get("reward_per_fire", -1.0)
        self._reward_per_move = kwargs.get("reward_per_move", -1.0)
        self._reward_reach = kwargs.get("reward_reach", 10)
        self._reward_burned = kwargs.get("reward_per_move", -10.0)# grandee
        

        # Cells
        self._empty = 0
        self._tree = 1
        self._fire = 2

        # Env Behavior Parameters

        self._p_fire = kwargs.get("p_fire", 0.033)
        self._p_tree = kwargs.get("p_tree", 0.033)

        self._effects = {self._fire: self._empty}

        scale = (nrows + ncols) // 2
        #self._max_freeze = int(speed * scale) if freeze is None else freeze
        self._max_freeze = int(0.3 * scale) if freeze is None else freeze
        # For `MoveModify`
        self._action_sets = {
            "up": {up_left, up, up_right},
            "down": {down_left, down, down_right},
            "left": {up_left, left, down_left},
            "right": {up_right, right, down_right},
            "not_move": {not_move},
        }

        self._set_spaces()

        self.cellular_automaton = ForestFire(
            self._empty, self._tree, self._fire, **self.ca_space
        )

        self.move = Move(self._action_sets, **self.move_space)
        self.modify = Modify(self._effects, **self.modify_space)

        self.move_modify = MoveModify(self.move, self.modify, **self.move_modify_space)

        # Composite Operators
        self._SARS = SARS(
            self.cellular_automaton,
            self.move_modify,
            self._max_freeze,
            **self.SARS_space,
        )

    # Gym API
    # step, reset & seed methods inherited from parent class

    def render(self, mode="human"):
        return render(self)

    def _award(self):
        ncells = self.nrows * self.ncols
        
        dict_counts = self.count_cells(self.grid)

        cell_counts = np.array(
            [dict_counts[self._empty], dict_counts[self._tree], dict_counts[self._fire]]
        )

        cell_counts_relative = cell_counts / ncells

        #reward_weights = np.array(
        #[self._reward_per_empty, self._reward_per_tree, self._reward_per_fire]
        #)
        #reward_weights = np.array([self._reward_per_move, self._reward_burned])
        #print(self._reward_per_move)
        reward_weights = self._reward_per_move + self._reward_burned
        #print(f'reward_weights {reward_weights}')

        #return np.dot(reward_weights, cell_counts_relative)
        return reward_weights

    def _is_done(self):
        _, position, position_goal, _ = self.context
        #print(f'position {position} position goal {position_goal}')
        if  (position == position_goal).all():
            print(f'pos {position} and {position_goal}position final')
            return True
        else:
            return False
        
    def _is_burned(self):
        _, position, _, _ = self.context
        
        value_grid_agent = self.grid[tuple(position)]
        
        if  value_grid_agent == 2:
            return True
        else:
            return False

    def _report(self):
        return {"hit": self.modify.hit}

    def _set_spaces(self):
        self.ca_params_space = spaces.Box(0.0, 1.0, shape=(2,))
        self.position_space = spaces.MultiDiscrete([self.nrows, self.ncols])
        self.position_goal_space = spaces.MultiDiscrete([self.nrows, self.ncols])
        self.freeze_space = spaces.Discrete(self._max_freeze + 1)

        self.context_space = spaces.Tuple(
            (self.ca_params_space, self.position_space, 
             self.position_goal_space, self.freeze_space)
        )

        self.grid_space = GridSpace(
            values=[self._empty, self._tree, self._fire],
            shape=(self.nrows, self.ncols),
        )

        # RL spaces

        self.action_space = spaces.Discrete(self._n_actions)
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        # Suboperators Spaces

        self.ca_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.ca_params_space,
        }

        self.move_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.position_space,
        }

        self.modify_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Discrete(2),
            "context_space": self.position_space,
        }

        self.move_modify_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Tuple((self.action_space, spaces.Discrete(2))),
            "context_space": self.position_space,
        }

        self.SARS_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }

class SARS(Operator):
    from collections import namedtuple

    Suboperators = namedtuple("Suboperators", ["cellular_automaton", "move_modify"])

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(self, cellular_automaton, move_modify, max_freeze, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.move_modify = move_modify
        self.ca = cellular_automaton

        self.suboperators = self.Suboperators(cellular_automaton, move_modify)

        self.max_freeze = max_freeze
        self.freeze_space = spaces.Discrete(max_freeze + 1)

    def update(self, grid, action, context):
        ca_params, position, position_goal, freeze = context

        if freeze == 0:
            grid, ca_params = self.ca(grid, None, ca_params)
            grid, position = self.move_modify(grid, (action, True), position)

            freeze = np.array(self.max_freeze)

        else:
            grid, position = self.move_modify(grid, (action, True), position)

            freeze = np.array(freeze - 1)

        context = ca_params, position, position_goal, freeze

        return grid, context
    
def sarsa(env, episodes=200, alpha=0.5, gamma=1.0, epsilon=0.1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    def choose_action(state):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        return np.argmax(Q[state])

    for episode in range(episodes):
        state, _ = env.reset()
        action = choose_action(state)
        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = choose_action(next_state)

            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action
    return Q
"""
class SARS(Operator):
    from collections import namedtuple

    Suboperators = namedtuple("Suboperators", ["cellular_automaton", "move_modify"])

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(self, cellular_automaton, move_modify, max_freeze, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.move_modify = move_modify
        self.ca = cellular_automaton

        self.suboperators = self.Suboperators(cellular_automaton, move_modify)

        self.max_freeze = max_freeze
        self.freeze_space = spaces.Discrete(max_freeze + 1)

    def update(self, grid, action, context):
        ca_params, position, freeze = context

        if freeze == 0:
            grid, ca_params = self.ca(grid, None, ca_params)
            #grid, position = self.move_modify(grid, (action, True), position)

            freeze = np.array(self.max_freeze)

        else:
            grid, position = self.move_modify(grid, (action, True), position)

            freeze = np.array(freeze - 1)

        context = ca_params, position, freeze

        return grid, context
"""