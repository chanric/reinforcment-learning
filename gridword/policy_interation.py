from GridWorld import negative_grid, print_values, print_policy
import numpy as np

SMALL_ENOUGH = 1e-3 #convergence threshold
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


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



def check_if_policy_converges(grid, V, policy):
    is_policy_converged = True
    for s in grid.all_states:
        if s in policy:
            #first check is if this is a state to example
            old_best_action = policy[s]
            curr_best_value = float('-inf')
            best_action = None
            for a in grid.location_to_action[s]:
                r = grid.get_reward(s, a)
                value = r + GAMMA*V[grid.current_state]
                if curr_best_value < value:
                    curr_best_value = value
                    best_action = a

            if best_action != old_best_action:
                is_policy_converged = False
                policy[s] = best_action

    return is_policy_converged

def evalulate_v_for_policy(policy, grid, V):
    # evaluate policy
    while True:
        biggest_change = 0

        for s in grid.all_states:
            old_v = V[s]

            if s in policy:
                # pick action in this case. we have fix policy
                r = grid.get_reward(s, policy[s])
                V[s] = r + GAMMA * V[grid.current_state]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break

if __name__ == '__main__':
    grid = negative_grid(-0.3)

    print("rewards:")
    print_values(grid.location_to_rewards, grid)

    # intialize random policy. then update
    policy = {}
    for s in grid.location_to_action.keys():
        policy[s] = np.random.choice(grid.location_to_action[s])

    print("initial policy:")
    print_policy(policy, grid)

    V = initalize_V(grid)
    while True:
        #evaluate policy to find V
        evalulate_v_for_policy(policy, grid, V)

        """
        Summary: change policy for biggest V
        Look at the policy. Now that we know V for the policy, see if we can update for more V 
        """
        if check_if_policy_converges(grid, V, policy):
            break



    print("finished")
    print_values(V, grid)

    print("best policy")
    print_policy(policy, grid)
