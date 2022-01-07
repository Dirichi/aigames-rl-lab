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
        slip_states = [self.apply_action(state, act) for act in range(self.n_actions)]
        slip_probability = self.slip / len(slip_states)
        for slip_state in slip_states:
          state_probability_map[slip_state] = state_probability_map.get(slip_state, 0) + slip_probability

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
      if state == self.absorbing_state:
        return ''
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

def test_transition_probs(test_env):
    test_probs = np.load("p.npy")
    env_probs = np.ones(test_probs.shape)
    for next_state in range(test_probs.shape[0]):
        for state in range(test_probs.shape[1]):
            for action in range(test_probs.shape[2]):
                env_probs[next_state][state][action] = test_env.p(next_state, state, action)

    failures = env_probs != test_probs
    num_failures = sum(failures.reshape(-1))
    assert num_failures == 0, "# of discrepancies in transition probability: {0}".format(num_failures)

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
        print('Reward: {0}.'.format(r))


def epsilon_greedy_action(action_values, epsilon, random_state):
  if len(set(action_values)) == 1:
    return random_state.randint(len(action_values))

  if random_state.random_sample() > epsilon:
    return action_values.argmax()

  return random_state.randint(len(action_values))


################ Model-based algorithms helpers ################

def trasition_value(env, next_state, state, action, values, gamma):
  prob_next_state = env.p(next_state, state, action)
  reward = env.r(next_state, state, action)
  next_state_value = values[next_state]

  return  prob_next_state * (reward + (gamma * next_state_value))

def state_action_value(env, policy, state, action, values, gamma):
  total = 0
  for next_state in range(env.n_states):
    prob_action = policy[state] == action
    value = trasition_value(env, next_state, state, action, values, gamma)
    total += (prob_action * value)

  return total

def state_value(env, policy, state, values, gamma):
  total = 0
  for action in range(env.n_actions):
    action_value = state_action_value(env, policy, state, action, values, gamma)
    total += action_value

  return total

################ Model-based algorithms ################

# POLICY EVALUATION
def policy_evaluation(env, policy, gamma, theta, max_iterations):
  values = np.zeros(env.n_states, dtype=np.float)

  for _ in range(max_iterations):
    max_delta = 0
    for state in range(env.n_states):
      initial_value = values[state]
      values[state] = state_value(env, policy, state, values, gamma)
      delta = abs(values[state] - initial_value)
      max_delta = max(max_delta, delta)
    if max_delta < theta:
      break

  return values

# POLICY IMPROVEMENT
def policy_improvement(env, values, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    for state in range(env.n_states):
      action_values = np.zeros(env.n_actions)
      for action in range(env.n_actions):
        action_value = 0
        for next_state in range(env.n_states):
          action_value += trasition_value(env, next_state, state, action, values, gamma)

        action_values[action] = action_value
      policy[state] = action_values.argmax()


    return policy

# POLICY ITERATION
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    total_iterations = 0

    values = np.zeros(env.n_states, dtype=np.float)
    for _ in range(max_iterations):
        total_iterations += 1
        previous_policy = policy
        values = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, values, gamma)
        if (previous_policy == policy).all():
            break

    print("Policy Iteration Num. Iterations: {0}".format(total_iterations))
    return policy, values

# VALUE ITERATION
def value_iteration(env, gamma, theta, max_iterations, values=None):
    if values is None:
        values = np.zeros(env.n_states)
    else:
        values = np.array(values, dtype=np.float)
    total_iterations = 0

    for _ in range(max_iterations):
      total_iterations += 1
      delta = 0
      for state in range(env.n_states):
        old_value = values[state]
        action_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
          action_value = 0
          for next_state in range(env.n_states):
            action_value += trasition_value(env, next_state, state, action, values, gamma)
          action_values[action] = action_value
        values[state] = action_values.max()
        delta = max(delta, abs(values[state] - old_value))

      if delta < theta:
        break

    policy = np.zeros(env.n_states, dtype=int)

    for state in range(env.n_states):
      action_values = np.zeros(env.n_actions)
      for action in range(env.n_actions):
        action_value = 0
        for next_state in range(env.n_states):
          action_value += trasition_value(env, next_state, state, action, values, gamma)
        action_values[action] = action_value
      policy[state] = action_values.argmax()

    print("Value Iteration Num. Iterations: {0}".format(total_iterations))
    return policy, values

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        state = env.reset()
        done = False

        action = epsilon_greedy_action(q[state], epsilon[i], random_state)

        while not done:
          next_state, reward, done = env.step(action)
          next_action = epsilon_greedy_action(q[next_state], epsilon[i], random_state)
          target_q_value = reward + (gamma * q[next_state][next_action])
          temporal_difference = target_q_value - q[state][action]
          q[state][action] = q[state][action] + (eta[i] * temporal_difference)

          state = next_state
          action = next_action

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def sarsa_convergence_test(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    theta = 0.001
    max_iterations = 100
    _, optimum_policy_values = policy_iteration(env, gamma, theta, max_iterations)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        state = env.reset()
        done = False

        action = epsilon_greedy_action(q[state], epsilon[i], random_state)

        while not done:
          next_state, reward, done = env.step(action)
          next_action = epsilon_greedy_action(q[next_state], epsilon[i], random_state)
          target_q_value = reward + (gamma * q[next_state][next_action])
          temporal_difference = target_q_value - q[state][action]
          q[state][action] = q[state][action] + (eta[i] * temporal_difference)

          state = next_state
          action = next_action
        if i % 100 == 0:
          policy = q.argmax(axis=1)
          current_policy_values = policy_evaluation(env, policy, gamma, theta, max_iterations)
          max_delta = (optimum_policy_values - current_policy_values).max()
          print("Max delta at Episode {0}: {1}".format(i, max_delta))

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        state = env.reset()
        done = False

        while not done:
          action = epsilon_greedy_action(q[state], epsilon[i], random_state)
          next_state, reward, done = env.step(action)
          temporal_difference = reward + (gamma * q[next_state].max()) - q[state][action]
          q[state][action] = q[state][action] + (eta[i] * temporal_difference)

          state = next_state

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def q_learning_convergence_test(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    theta = 0.001
    max_iterations = 100
    iteration_at_convergence = None
    max_delta_at_convergence = None
    _, optimum_policy_values = policy_iteration(env, gamma, theta, max_iterations)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    policy = q.argmax(axis=1)

    for i in range(max_episodes):
        state = env.reset()
        done = False

        while not done:
          action = epsilon_greedy_action(q[state], epsilon[i], random_state)
          next_state, reward, done = env.step(action)
          temporal_difference = reward + (gamma * q[next_state].max()) - q[state][action]
          q[state][action] = q[state][action] + (eta[i] * temporal_difference)

          state = next_state
        if i % 100 == 0:
          policy = q.argmax(axis=1)
          current_policy_values = policy_evaluation(env, policy, gamma, theta, max_iterations)
          max_delta = (optimum_policy_values - current_policy_values).max()
          if max_delta != max_delta_at_convergence:
            max_delta_at_convergence = max_delta
            iteration_at_convergence = i

    return iteration_at_convergence, max_delta_at_convergence

################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)

def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        done = False
        q = features.dot(theta)
        action = epsilon_greedy_action(q, epsilon[i], random_state)

        while not done:
          q = features.dot(theta)
          next_features, reward, done = env.step(action)
          next_q = next_features.dot(theta)
          next_action = epsilon_greedy_action(next_q, epsilon[i], random_state)
          delta = (reward + gamma * next_q[next_action]) - q[action]

          theta = theta + (eta[i] * delta * features[action])
          features = next_features
          action = next_action

    return theta

def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        done = False

        while not done:
            q = features.dot(theta)
            action = epsilon_greedy_action(q, epsilon[i], random_state)
            next_features, reward, done = env.step(action)
            next_q = next_features.dot(theta)
            delta = (reward + gamma * next_q.max()) - q[action]
            theta = theta + (eta[i] * delta * features[action])
            features = next_features

    return theta

################ Main function ################

def main():
    seed = 0

    # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    test_transition_probs(env)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning Convergence Test')
    convergence_episode, diff_from_optimum = q_learning_convergence_test(env, max_episodes, eta, gamma, epsilon, seed=seed)
    print("Converged at episode {0}, and diff from optimum: {1}".format(convergence_episode, diff_from_optimum))

    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

main()