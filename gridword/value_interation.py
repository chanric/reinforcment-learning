from GridWorld import negative_grid, print_values, print_policy, ALL_POSSIBLE_ACTIONS
from common_grid_funcs import initalize_V
import numpy as np

SMALL_ENOUGH = 1e-3 #convergence threshold
GAMMA = 0.9





def generate_policy_from_V(grid, V):
    policy = {}
    for s in grid.location_to_action.keys():
        best_action = None
        best_V = float('-inf')
        for a in grid.location_to_action[s]:
            r = grid.get_reward(s, a)
            v = r + GAMMA*V[grid.current_state]
            if v > best_V:
                best_V = v
                best_action = a
        policy[s] = best_action
    return policy



def compute_V_by_running_through_all_actions(grid, V):
    states_with_actions = grid.location_to_action.keys()

    while True:
        biggest_change = 0

        for s in grid.all_states:
            old_v = V[s]

            if s in states_with_actions:
                best_v = float('-inf')

                for a in ALL_POSSIBLE_ACTIONS:
                    r = grid.get_reward(s, a)
                    curr_v = r + GAMMA*V[grid.current_state]
                    if curr_v > best_v:
                        best_v = curr_v
                V[s] = best_v

            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break

if __name__ == '__main__':
    grid = negative_grid(-1)

    print("rewards:")
    print_values(grid.location_to_rewards, grid)

    V = initalize_V(grid)

    compute_V_by_running_through_all_actions(grid, V)

    """
    Summary: given V, find out the best action for each sate 
    Note.  we are actually generating a policy
    """
    policy = generate_policy_from_V(grid, V)

    print("finished")
    print_values(V, grid)

    print("best policy")
    print_policy(policy, grid)
