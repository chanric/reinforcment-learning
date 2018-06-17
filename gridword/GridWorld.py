
class GridWorld:

    def __init__(self, width, height, starting_state, location_to_rewards, location_to_action):
        self.width = width
        self.height = height;
        self.i = starting_state[0]
        self.j = starting_state[1]
        self.location_to_rewards = location_to_rewards
        self.location_to_action = location_to_action

        self.all_states = set(self.location_to_action.keys() | self.location_to_rewards.keys())

    def set_state(self, s):
        if len(s) != 2:
            raise Exception("we expect %s to be a (x,y)" % s)
        self.i = s[0]
        self.j = s[1]

    @property
    def current_state(self):
        return self.i, self.j

    def is_terminal(self, s):
        return s not in self.location_to_action


    def get_reward(self, init_state, action):
        self.set_state(init_state)
        return self.move(action)

    def move(self, action):
        if action in self.location_to_action[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
            else:
                raise Exception("action must be one of (U, D, R, L)")
        #return award for new state if any
        return self.location_to_rewards.get((self.i, self.j), 0)


    def undo_move(self, action):
        if action in self.location_to_action[(self.i, self.j)]:
            if action == 'U':
                self.i += 1
            elif action == 'D':
                self.i -= 1
            elif action == 'R':
                self.j -= 1
            elif action == 'L':
                self.j += 1
            else:
                raise Exception("action must be one of (U, D, R, L)")
    def is_game_over(self):
        return (self.i, self.j) not in self.location_to_action

    def all_states(self):
        return set(self.location_to_action.keys() | self.location_to_rewards.keys())


def print_values(V, g):
    for i in range(g.width):
        print("---------------------------")
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    for i in range(g.width):
        print("---------------------------")
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")


def print_policy(P, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")

ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

__STANDARD_ACTIONS = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }



def standard_grid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
  rewards = {(0, 3): 1, (1, 3): -1}
  g = GridWorld(3, 4, (2, 0), rewards, __STANDARD_ACTIONS)
  return g


def negative_grid(step_cost=-0.1):
    rewards = {
        (0, 3): 1,
        (1, 3): -1,
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    }
    return GridWorld(3, 4, (2, 0), rewards, __STANDARD_ACTIONS)
