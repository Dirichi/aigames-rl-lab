import numpy as np

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
  for _ in range(max_iterations):
    previous_policy = policy
    values = policy_evaluation(env, policy, gamma, theta, max_iterations)
    policy = policy_improvement(env, policy, values, gamma)
    if (previous_policy == policy).all():
      break

  return policy

# VALUE ITERATION
def value_iteration(env, gamma, theta, max_iterations):
  policy = np.zeros(env.n_states, dtype=np.int)
  value = np.zeros(env.n_states, dtype=np.float)
  for _ in range(max_iterations):
    for state in range(env.n_states):
      pass

  return policy, value






