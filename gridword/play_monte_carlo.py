from GridWorld import standard_grid, print_values, print_policy
import numpy as np

GAMMA = 0.9

"""
With monto carlo. we play the game with a random start state.  Then get the results to interprete

"""


def play_game_and_get_returns(starting_state, grid, policy):
    """
    :param starting_state: starting point
    :param grid:
    :param policy:
    :return: list_of_state_to_returns (if state is seen 2x, only first one is calculated)
    """

    state_rewards = [(starting_state, 0)]
    grid.set_state(starting_state)

    while not grid.is_game_over():
        s = grid.current_state
        #move
        r = grid.get_reward(s, policy[s])
        state_rewards.append((grid.current_state, r))

    #Now back propagate to find awards
    G=0
    state_returns = []
    for s, r in reversed(state_rewards):
        G = r + GAMMA*G
        state_returns.append((s,G))

    #throw out last element becuase it is a terminal state
    return list(reversed(state_returns))[:-1]



if __name__ == '__main__':
    #this demos how to evaluate a policy by playing the game.
    grid = standard_grid()
    V = {}
    #policy to check
    POLICY = {
        (2, 0): 'U',  (2, 1): 'R', (2, 2): 'R',   (2, 3): 'U',
        (1, 0): 'U',               (1, 2): 'U', #terminal L
        (0, 0): 'R',  (0, 1): 'R', (0, 2): 'R', #terminal win
    }
    print_policy(POLICY, grid)


    state_to_return_list = {}
    for s in grid.location_to_action.keys():
        state_to_return_list[s] = []


    start_state_list = list(grid.location_to_action.keys())
    for i in range(1000):
        #run game
        test_state_index = np.random.choice(len(start_state_list))
        state_returns = play_game_and_get_returns(start_state_list[test_state_index], grid, POLICY)

        eval_states = set()
        for s, G in state_returns:
            if s not in eval_states:
                eval_states.add(s)
                state_to_return_list[s].append(G)
                V[s] = np.mean(state_to_return_list[s])

    print_values(V, grid)
