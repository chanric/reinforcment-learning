from GridWorld import standard_grid, print_values, print_policy, negative_grid, ALL_POSSIBLE_ACTIONS
import numpy as np
from common_grid_funcs import intitalize_Q_and_returns, action_with_epsilon_greedy, get_best_action_from_q
import matplotlib.pyplot as plt


GAMMA = 0.9
ALPHA = 0.1


"""
In saras we play the game.  Find the best action.  Apply chance for random action, Then update with
      Q[s][a] = Q[s][a] + alpha*(r + GAMMA*Q[s2][a2] - Q[s][a])
      where a2 is the next action ( might not be the best)
      Adjust update count (since alpha is decaying with count
"""


if __name__ == '__main__':
    grid = negative_grid()

    Q, _ = intitalize_Q_and_returns(grid)
    #add in terminal states as zero
    for s in grid.all_states:
        if s not in grid.location_to_action.keys():
            Q[s] = {a: 0 for a in ALL_POSSIBLE_ACTIONS}
    """
    Q = {}
    states = grid.all_states
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
            """

    #init count to track total updates (for logging only)
    logging_update_counts= {}
    #init count to keep update for s given a
    update_counts_sa = {s: {a: 1.0 for a in ALL_POSSIBLE_ACTIONS} for s in grid.all_states}

    t = 1.0
    logging_deltas = []
    for i in range(10000):
        if i % 100 == 0:
            t += 1e-2

        s = (2,0) #fix start state
        grid.set_state(s)
        a = get_best_action_from_q(Q, s, grid)
        a = action_with_epsilon_greedy(a, eps=0.5/t)
        log_biggest_change = 0
        while not grid.is_game_over():
            r = grid.get_reward(s, a)
            s2 = grid.current_state

            a2 = get_best_action_from_q(Q, s2, grid)
            a2 = action_with_epsilon_greedy(a2, eps=0.5/t)

            alpha = ALPHA/update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            log_old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + alpha*(r + GAMMA*Q[s2][a2] - Q[s][a])

            s = s2
            a = a2

            #for logging only
            logging_update_counts[s] = logging_update_counts.get(s,0) + 1
            log_biggest_change = max(log_biggest_change, np.abs(log_old_qsa - Q[s][a]))
        logging_deltas.append(log_biggest_change)

    plt.plot(logging_deltas)
    plt.show()

    policy = {}
    V = {}
    for s in grid.location_to_action.keys():
        policy[s] = get_best_action_from_q(Q, s, grid)
        V[s] = Q[s][policy[s]]

    # what's the proportion of time we spend updating each part of Q?
    print("update counts:")
    total = np.sum(list(logging_update_counts.values()))
    for k, v in logging_update_counts.items():
      logging_update_counts[k] = float(v) / total
    print_values(logging_update_counts, grid)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)





