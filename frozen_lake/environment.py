################ Environment ################

import numpy as np
import contextlib

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.width = len(self.lake[0])
        self.height = len(self.lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1
        self.transition_probabilities = {}
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        """
        Returns the probability of transitioning from state to next_state
        after taking the given action in state.
        """
        transition = (next_state, state, action)
        key = ",".join([str(el) for el in transition])
        if key not in self.transition_probabilities:
          prob = self.transition_probability(next_state, state, action)
          self.transition_probabilities[key] = prob

        return self.transition_probabilities[key]

    def transition_probability(self, next_state, state, action):
        state_probability_map = {}
        slip_states = set([self.apply_action(state, act) for act in range(self.n_actions)])
        slip_probability = self.slip / len(slip_states)
        for slip_state in slip_states:
          state_probability_map[slip_state] = slip_probability

        expected_state = self.apply_action(state, action)
        state_probability_map[expected_state] += (1 - self.slip)
        return state_probability_map.get(next_state, 0)



    def r(self, next_state, state, action):
      if self.is_goal(state):
        return 1

      return 0

    def apply_action(self, state, action):
      """
      Returns the expected next state after applying an action
      in a state
      """
      if state == self.absorbing_state:
        return state

      if self.is_goal_or_hole(state):
        return self.absorbing_state

      state_coords = self.state_to_coords(state)
      action_coords = self.action_to_coords(action)

      next_state_coords = [
        state_coords[0] + action_coords[0],
        state_coords[1] + action_coords[1]
      ]

      next_state = self.coords_to_state(next_state_coords)
      return next_state if self.valid_coords(next_state_coords) else state

    def is_goal_or_hole(self, state):
      return self.is_goal(state) or self.is_hole(state)

    def is_goal(self, state):
      return self.state_value(state) == "$"

    def is_hole(self, state):
      return self.state_value(state) == "#"

    def state_to_coords(self, state):
      assert state < len(self.lake_flat), "{0} can not be represented in coordinates".format(state)

      x_idx = state % self.width
      y_idx = (state - x_idx) / self.width
      return [x_idx, y_idx]

    def state_value(self, state):
      return self.lake_flat[int(state)]

    def coords_to_state(self, coords):
      return (coords[1] * self.width) + coords[0]

    def valid_coords(self, coords):
      if (coords[0] < 0) or (coords[0] >= self.width):
        return False

      if (coords[1] < 0) or (coords[1] >= self.height):
        return False

      return True

    def action_to_coords(self, action):
      if action == 0: #UP
        return [0, -1]

      if action == 1: #LEFT
        return [-1, 0]

      if action == 2: #DOWN
        return [0, 1]

      if action == 3: #RIGHT
        return [1, 0]

      return [0, 0]


    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('State: {0}, Reward: {1}.'.format(state, r))



lake =  [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

env = FrozenLake(lake, 0.1, 16)
play(env)