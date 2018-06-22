from GridWorld import standard_grid, print_values, print_policy, negative_grid
import numpy as np
from common_grid_funcs import initalize_V, action_with_epsilon_greedy
import matplotlib.pyplot as plt

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1


"""
Shows TD0 prediction.
Unlike SARSA, we are using V[s] not Q[s][a]
Main points are
    1.  play game & return [] of (s,r)
    2.  use  next state to update V[S] with
         V[s] = V[s] + ALPHA*(r + GAMMA*V[s2] - V[s])
"""
__START_STATE = (2,0)

def play_game_return_state_rewards(grid, policy, starting_state=__START_STATE):
    """
    :param grid:
    :param policy:
    :param starting_state:
    :return: [] of (s, r)
    """
    s = starting_state
    state_rewards = [(s,0)]
    #set state to init game
    grid.set_state(s)
    while not grid.is_game_over():
        r = grid.get_reward(s, action_with_epsilon_greedy(policy[s]))
        s = grid.current_state
        state_rewards.append((s, r))

    return state_rewards





if __name__ == '__main__':
    grid = standard_grid()

    POLICY_TO_EVAL = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    V = initalize_V(grid)


    for i in range(1000):
        #play game
        state_to_returns = play_game_return_state_rewards(grid, POLICY_TO_EVAL) #play_game_return_state_rewards(grid, POLICY_TO_EVAL)
        #calc V
        for j in range(len(state_to_returns) -  1):
            s, r0 = state_to_returns[j]
            s2, r = state_to_returns[j+1]
            V[s] = V[s] + ALPHA * (r + GAMMA * V[s2] - V[s])
            #V[s] = V[s] + ALPHA*(r + GAMMA*V[s2] - V[s])

    print("rewards")
    print_values(grid.location_to_rewards, grid)
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(POLICY_TO_EVAL, grid)




