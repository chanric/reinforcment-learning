import numpy as np
from GridWorld import ALL_POSSIBLE_ACTIONS

def initalize_V(grid):
    V = {}
    for s in grid.all_states:
        if s in grid.location_to_action:
            # set random
            V[s] = np.random.random()
        else:
            # terminal state
            V[s] = 0
    return V


def intitalize_Q_and_returns(grid):
    """
    :param grid:
    :return: for each s, Q[s] with empty dict.  returns[s] = {a:[]) for each a
    """
    Q = {}
    returns = {}
    action_states = grid.location_to_action.keys()
    for s in action_states:
        Q[s] = dict([(a,-3) for a in ALL_POSSIBLE_ACTIONS])
        returns[s] = dict([(a,[]) for a in ALL_POSSIBLE_ACTIONS])
    return Q, returns


def action_with_epsilon_greedy(action, possible_actions=ALL_POSSIBLE_ACTIONS, eps=0.1):
    p = np.random.random()
    if p < (1-eps):
        return action
    else:
        return np.random.choice(possible_actions)
