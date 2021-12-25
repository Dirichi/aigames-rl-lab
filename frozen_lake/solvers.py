import numpy as np

def epsilon_greedy_action(q, state, epsilon, random_state):
  if len(set(q[state])) == 1:
    return random_state.randint(len(q[state]))

  if random_state.random_sample() > epsilon:
    return q[state].argmax()

  return random_state.randint(len(q[state]))

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # q = random_state.rand(env.n_states, env.n_actions) * 0.001
    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        state = env.reset()
        done = False

        action = epsilon_greedy_action(q, state, epsilon[i], random_state)

        while not done:
          next_state, reward, done = env.step(action)
          next_action = epsilon_greedy_action(q, next_state, epsilon[i], random_state)
          target_q_value = reward + (gamma * q[next_state][next_action])
          temporal_difference = target_q_value - q[state][action]
          q[state][action] = q[state][action] + (eta[i] * temporal_difference)

          state = next_state
          action = next_action

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
          action = epsilon_greedy_action(q, state, epsilon[i], random_state)
          next_state, reward, done = env.step(action)
          temporal_difference = reward + (gamma * q[next_state].max()) - q[state][action]
          q[state][action] = q[state][action] + (eta[i] * temporal_difference)

          state = next_state

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        # TODO:

    return theta

def arg_max(args_list, func):
  arg_values = [func.__call__(argument) for argument in args_list]
  max_arg_value = max(arg_values)
  max_index = arg_values.index(max_arg_value)
  return args_list[max_index]


def calculate_state_action_value(env, policy, state, action, values, gamma):
  state_action_value = 0
  for next_state in range(env.n_states):
    transition_value = calculate_trasition_value(env, policy, next_state, state, action, values, gamma)
    state_action_value += transition_value

  return state_action_value


def calculate_state_value(env, policy, state, values, gamma):
  state_value = 0
  for action in range(env.n_actions):
    state_action_value = calculate_state_action_value(env, policy, state, action, values, gamma)
    state_value += state_action_value

  return state_value

def calculate_q_value(env, policy, state, action, values, gamma):
  policy_prime = policy.copy()
  policy_prime[state] = action
  return calculate_state_value(env, policy_prime, state, values, gamma)

def parametrized_q_value(env, policy, state, values, gamma):
  def internal_func(action):
    return calculate_q_value(env, policy, state, action, values, gamma)

  return internal_func


def calculate_trasition_value(env, policy, next_state, state, action, values, gamma):
  prob_action = policy[state] == action
  prob_next_state = env.p(next_state, state, action)
  reward = env.r(next_state, state, action)
  next_state_value = values[next_state]

  return prob_action * prob_next_state * (reward + (gamma * next_state_value))


# POLICY EVALUATION
def policy_evaluation(env, policy, gamma, theta, max_iterations):
  values = np.zeros(env.n_states, dtype=np.float)

  for _ in range(max_iterations):
    max_delta = 0
    for state in range(env.n_states):
      initial_value = values[state]
      values[state] = calculate_state_value(env, policy, state, values, gamma)
      delta = abs(values[state] - initial_value)
      max_delta = max(max_delta, delta)
    if max_delta < theta:
      break

  return values


# POLICY IMPROVEMENT
def policy_improvement(env, policy, values, gamma):
  improved_policy = np.zeros(env.n_states, dtype=np.int)
  for state in range(env.n_states):
    parametrized_q_func = parametrized_q_value(env, policy, state, values, gamma)
    arg_max_action = arg_max(list(range(env.n_actions)), parametrized_q_func)
    improved_policy[state] = arg_max_action

  return improved_policy


# POLICY ITERATION
def policy_iteration(env, gamma, theta, max_iterations):
  policy = np.zeros(env.n_states, dtype=np.int)
  values = np.zeros(env.n_states, dtype=np.float)
  for _ in range(max_iterations):
    previous_policy = policy
    values = policy_evaluation(env, policy, gamma, theta, max_iterations)
    policy = policy_improvement(env, policy, values, gamma)
    if (previous_policy == policy).all():
      break

  return policy, values

# VALUE ITERATION
def value_iteration(env, gamma, theta, max_iterations):
  policy = np.zeros(env.n_states, dtype=np.int)
  value = np.zeros(env.n_states, dtype=np.float)
  for _ in range(max_iterations):
    for state in range(env.n_states):
      pass

  return policy, value







