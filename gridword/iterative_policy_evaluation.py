from GridWorld import standard_grid, print_values
import numpy as np

SMALL_ENOUGH = 1e-3 #convergence threshold





if __name__ == '__main__':

    grid = standard_grid()
    states = grid.all_states

    #V for value
    V = {}
    for s in states:
        V[s] = 0

    gamma=1.0

    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]

            if s in grid.location_to_action:
                # calculate new value.  given chance of each possible move form state/location
                new_v = 0;

                p_a = 1.0 / len(grid.location_to_action[s])
                for a in grid.location_to_action[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v-V[s]))

        if biggest_change < SMALL_ENOUGH:
            break

    print_values(V, grid)


