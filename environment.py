import numpy as np
import math
import policy

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
        p = [self.p(next_state, state, action) for next_state in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, dist, seed=None):
        super().__init__(n_states=n_states, n_actions=n_actions, seed=seed)

        self.max_steps = max_steps
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1 / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done


class GridWorld(Environment):
  def __init__(
    self,
    width,
    height,
    max_steps,
    wall_indexes,
    goal_states,
    trap_states,
    seed=None):
      self.width = width
      self.height = height
      self.grid_size = (width * height)
      self.wall_indexes = set(wall_indexes)
      self.goal_states = set(goal_states)
      self.trap_states = set(trap_states)
      self.preterminal_states = set(goal_states + trap_states)
      n_states = self.grid_size + 1 # An additional terminal state
      dist = np.full(n_states, 1 / self.grid_size)
      dist[-1] = 0 # Probability of starting from the terminal state is 0
      super().__init__(n_states=n_states, n_actions=4, max_steps=max_steps, dist=dist, seed=seed)

  def p(self, next_state, state, action):
    """
    Returns the probability of transitioning from state to next_state
    after taking the given action in state.
    """
    expected_state = self.apply_action(state, action)
    return expected_state == next_state

  def r(self, next_state, state, action):
    if state in self.goal_states:
      return 1

    if state in self.trap_states:
      return -1

    return 0

  def apply_action(self, state, action):
    """
    Returns the expected next state after applying an action
    in a state
    """
    if self.is_terminal(state):
      return state

    if state in self.preterminal_states:
      return self.terminal_state()

    state_coords = self.state_to_coords(state)
    action_coords = self.action_to_coords(action)

    next_state_coords = [
      state_coords[0] + action_coords[0],
      state_coords[1] + action_coords[1]
    ]

    next_state = self.coords_to_state(next_state_coords)
    return next_state if self.valid_coords(next_state_coords) else state

  def valid_coords(self, coords):
    if (coords[0] < 0) or (coords[0] >= self.width):
      return False

    if (coords[1] < 0) or (coords[1] >= self.height):
      return False

    coords_state = self.coords_to_state(coords)
    return coords_state not in self.wall_indexes

  def state_to_coords(self, state):
    assert state < self.grid_size, "{0} can not be represented in coordinates".format(state)

    x_idx = state % self.width
    y_idx = (state - x_idx) / self.width
    return [x_idx, y_idx]

  def coords_to_state(self, coords):
    return (coords[1] * self.width) + coords[0]

  def action_to_coords(self, action):
    if action == 0: #UP
      return [0, 1]

    if action == 1: #RIGHT
      return [1, 0]

    if action == 2: #DOWN
      return [0, -1]

    if action == 3: #LEFT
      return [-1, 0]

    return [0, 0]

  def is_terminal(self, state):
    return state == self.terminal_state()

  def terminal_state(self):
    return self.grid_size

  def set_renderer(self, renderer):
    self.renderer = renderer

  def render(self):
    assert self.renderer, "No renderer available"
    self.renderer \
      .reset() \
      .set_show_states(True) \
      .set_show_current_state(True) \
      .set_show_goals(True) \
      .set_show_traps(True) \
      .set_show_walls(True) \
      .render(self)

class GridWorldRenderer:
  def __init__(self) -> None:
    self.cell_width = 20
    self.cell_height = 8
    self.cell_text_map = {}
    self.show_current_state = False
    self.show_states = False
    self.show_goals = False
    self.show_traps = False
    self.show_walls = False

  def add_cell_text(self, cell_id, cell_y, text):
    text_key = str(cell_id) + "@" + str(cell_y)
    self.cell_text_map[text_key] = text

  def set_show_states(self, show):
    self.show_states = show
    return self

  def set_show_current_state(self, show):
    self.show_current_state = show
    return self

  def set_show_goals(self, show):
    self.show_goals = show
    return self

  def set_show_traps(self, show):
    self.show_traps = show
    return self

  def set_show_walls(self, show):
    self.show_walls = show
    return self

  def reset(self):
    self.cell_text_map = {}
    self.show_current_state = False
    self.show_states = False
    self.show_goals = False
    self.show_traps = False
    self.show_walls = False
    return self

  def render(self, world):
    self.populate_default_cell_texts(world)

    for i in range(world.height):
      self.print_rowline(world)
      for j in range(self.cell_height):
        self.print_column_line(world, i, j)

    self.print_rowline(world)
    print(world.state)

  def populate_default_cell_texts(self, world):
    cell_mid_y = math.ceil(self.cell_height / 2)

    if self.show_states:
      for state in range(world.n_states):
        self.add_cell_text(state, cell_mid_y - 1, str(state))

    if self.show_goals:
      for state in world.goal_states:
        self.add_cell_text(state, cell_mid_y, "GOAL")

    if self.show_traps:
      for state in world.trap_states:
        self.add_cell_text(state, cell_mid_y, "TRAP")

    if self.show_walls:
      for state in world.wall_indexes:
        self.add_cell_text(state, cell_mid_y, "WALL")

    if self.show_current_state:
      self.add_cell_text(world.state, cell_mid_y + 1, "YOU ARE HERE")


  def print_rowline(self, world):
    row = ("-" * self.cell_width) * world.width
    print(row)

  def print_column_line(self, world, column_id, cell_y):
    columns = ""
    for row_id in range(world.width):
      val_to_print = self.print_cell_line(world, row_id, column_id, cell_y)

      columns += val_to_print
    print(columns)

  def print_cell_line(self, world, row_id, column_id, cell_y):
    cell_line = "|"  + (" " * (self.cell_width - 1))
    if row_id  == world.width - 1:
      cell_line += "|"

    col_id = (world.height - column_id - 1)
    cell_id = (col_id * world.width) + row_id

    cell_text_key = str(cell_id) + "@" + str(cell_y)
    cell_text = self.cell_text_map.get(cell_text_key, None)
    if cell_text:
      cell_line = self.insert(cell_line, cell_text)

    return cell_line

  def insert(self, string, value):
    string_arr = list(string)
    midpoint = len(string_arr) / 2
    val = str(value)
    start_point = math.floor(midpoint - (len(val) / 2))

    for i in range(len(val)):
      index = start_point + i
      string_arr[index] = val[i]

    return "".join(string_arr)

class PolicyRenderer():
  def __init__(self, world, world_renderer) -> None:
    self.world_renderer = world_renderer
    self.world = world

  def render(self, policy):
    cell_mid_y = math.ceil(self.world_renderer.cell_height / 2)
    self.world_renderer \
      .reset() \
      .set_show_states(True) \
      .set_show_goals(True) \
      .set_show_traps(True) \
      .set_show_walls(True)

    for state in range(self.world.n_states):
      action = self.action_to_str(policy[state])
      self.world_renderer.add_cell_text(state, cell_mid_y + 2, action)

    self.world_renderer.render(self.world)

  def action_to_str(self, action):
    if action == 0: #UP
      return "^"

    if action == 1: #RIGHT
      return "->"

    if action == 2: #DOWN
      return "v"

    if action == 3: #LEFT
      return "<-"

    return "UNKNOWN"




actions = ['w', 'd', 's', 'a']

env = GridWorld(width=4, height=3, max_steps=20, wall_indexes=[5], goal_states=[11], trap_states=[7])
renderer = GridWorldRenderer()
env.set_renderer(renderer)
env.reset()
env.render()
done = False

# while not done:
#   command = input('\nMove: ')
#   if command not in actions:
#     raise Exception('Invalid action')

#   state, r, done = env.step(actions.index(command))
#   env.render()
#   print("State: {0}, Reward:{1}".format(state, r))

opt_policy = policy.policy_iteration(env, gamma=0.8, theta=0.001, max_iterations=20)
print(opt_policy)
policy_renderer = PolicyRenderer(env, renderer)
policy_renderer.render(opt_policy)



