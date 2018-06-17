from GridWorld import standard_grid, print_values, print_policy, negative_grid
import numpy as np
from common_grid_funcs import intitalize_Q_and_returns, action_with_epsilon_greedy
import matplotlib.pyplot as plt
GAMMA = 0.9

"""
With monto carlo. we play the game with a fix random start state.  Then get the results to interpret
"""



def play_game_and_get_returns(starting_state, grid, policy):
    """
    Note: we return action.  This is useful if the policy is not always executed as planned (aka randomness)
    :param starting_state: starting point
    :param grid:
    :param policy:
    :return: list_of_state,actions_to_returns (if state is seen 2x, only first one is calculated)
    """

    state_rewards = [(starting_state, policy[starting_state], 0)]
    grid.set_state(starting_state)
    action = action_with_epsilon_greedy(policy[starting_state], grid.location_to_action[starting_state])
    while True:
        s = grid.current_state
        #move
        action = action_with_epsilon_greedy(policy[s], grid.location_to_action[s])
        r = grid.get_reward(s, action)
        if grid.is_game_over():
            state_rewards.append((s, None, r))
            break
        state_rewards.append((grid.current_state, action, r))


    #Now back propagate to find rewards
    G = state_rewards[-1][2]
    state_rewards = state_rewards[:-1] #drop last one because it represents a terminal state
    state_returns = []
    for s, a, r in reversed(state_rewards):
        G = r + (GAMMA * G)
        state_returns.append((s, a, G))

    #throw out last element becuase it is a terminal state
    return list(reversed(state_returns))


def play_game_and_get_returns2(starting_state, grid, policy):
    """
    Note: we return action.  This is useful if the policy is not always executed as planned (aka randomness)
    :param starting_state: starting point
    :param grid:
    :param policy:
    :return: list_of_state,actions_to_returns (if state is seen 2x, only first one is calculated)
    """

    s = (2, 0)
    grid.set_state(s)
    a = action_with_epsilon_greedy(policy[s])

    # be aware of the timing
    # each triple is s(t), a(t), r(t)
    # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
    states_actions_rewards = [(s, a, 0)]
    while True:
        r = grid.move(a)
        s = grid.current_state
        if grid.is_game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = action_with_epsilon_greedy(policy[s])  # the next state is stochastic
            states_actions_rewards.append((s, a, r))

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()  # we want it to be in order of state visited
    return states_actions_returns



if __name__ == '__main__':
    #this demos how to evaluate a policy by playing the game.
    grid = standard_grid()

    #init policy
    policy = {
        (0, 0): 'R', (0, 1): 'R', (0, 2): 'R',  # terminal win
        (1, 0): 'D',               (1, 2): 'L', #terminal L
        (2, 0): 'U', (2, 1): 'R', (2, 2): 'U', (2, 3): 'U',
    }

    for s in grid.location_to_action.keys():
        policy[s] = np.random.choice(grid.location_to_action[s])
    print_policy(policy, grid)




    Q, state_to_return_list = intitalize_Q_and_returns(grid)
    #START_STATE_LIST = list(grid.location_to_action.keys())
    q_change_diff_to_plot = []
    for i in range(5000):
        biggest_change=0 #optional
        #run game
        #test_state_index = np.random.choice(len(START_STATE_LIST))
        state_returns = play_game_and_get_returns2((0, 2), grid, policy)

        eval_states = set()
        for s, a, G in state_returns:
            if s not in eval_states:
                prev_Q = Q[s][a]  #optional for calculting biggest changes
                eval_states.add(s)
                state_to_return_list[s][a].append(G)
                Q[s][a] = np.mean(state_to_return_list[s][a])
                biggest_change = max(biggest_change, np.abs(Q[s][a] - prev_Q))


        #now pick the action with the best V for each state
        for s in policy:
            possible_actions = grid.location_to_action[s]
            curr_best_action = policy[s]
            curr_best_Q = float('-inf')
            for pa in possible_actions:
                if curr_best_Q < Q[s][pa]:
                    curr_best_Q = Q[s][pa]
                    curr_best_action = pa
            policy[s] = curr_best_action
        q_change_diff_to_plot.append(biggest_change)


    print(Q)
    print("NEW P")

    plt.plot(q_change_diff_to_plot)
    #plt.show()







    print_policy(policy, grid)
