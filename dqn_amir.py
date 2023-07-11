
from __future__ import print_function
import datetime, json, random



import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import PReLU

import matplotlib.pyplot as plt

import math



size_x_grid = 5
size_y_grid = 5

gbs1 = [1, 1]
gbs2 = [1, 3]
gbs3 = [3, 1]
gbs4 = [3, 3]
gbs5 = [0, 0]
gbs6 = [0,4]


GBS = [gbs1, gbs2, gbs3, gbs4,gbs5, gbs6]

gbs_range = 1
grid = np.zeros((size_x_grid, size_y_grid))


def calDistance(a, b, x, y):
    m = a - x
    n = b - y
    m = m ** 2
    n = n ** 2
    distance = m + n
    return math.sqrt(distance)


def in_range_gbs(i, j, GBS):
    for m in range(len(GBS)):
        if (calDistance(i, j, GBS[m][0], GBS[m][1]) <= gbs_range):
            return True
    return False


for i in range(size_x_grid):
    for j in range(size_y_grid):
        if (in_range_gbs(i, j, GBS)):
            grid[i][j] = 1
        else:
            grid[i][j] = 0

print(grid)



visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
UAV_mark1 = 0.5  # The current rat cell will be painting by gray 0.5

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
STAY = 4



# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
    STAY: 'stay',

}

num_actions = len(actions_dict)
print(num_actions)
# Exploration factor
epsilon = 0.1

# UAV_path is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a in_range_gbs cell, and 0.0 an out_of_range cell
# UAV1 = (row1, col1) initial UAV1 position (defaults to (0,0))
# UAV2 = (row2,col2) initial UAV2 position

UAV1_start = [0, 0]
UAV1_end = [4, 1]



class UAV(object):
    def __init__(self, grid, UAV1=(UAV1_start[0], UAV1_start[1])):
        self._grid = np.array(grid)
        nrows, ncols = self._grid.shape
        self.target1 = (UAV1_end[0], UAV1_end[1])
        self.in_range_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._grid[r, c] == 1.0]
        if self._grid[self.target1] == 0.0:
            raise Exception("Invalid grid: target cell must be in range of gbs")
        if not UAV1 in self.in_range_cells:
            raise Exception("Invalid UAV1 Location: must sit on a range of gbs cell")
        self.reset(UAV1)

    def reset(self, UAV1):
        self.UAV1 = UAV1
        self.grid = np.copy(self._grid)
        nrows, ncols = self.grid.shape
        row1, col1 = UAV1
        self.grid[row1, col1] = UAV_mark1
        self.state = (row1, col1, 'start')
        self.min_reward = -0.5 * self.grid.size #changed
        self.total_reward = 0
        self.visited1 = set()
        self.visited2 = set()

    def update_state(self, action):
        nrows, ncols = self.grid.shape
        nrow1, ncol1, nmode = UAV_row1, UAV_col1, mode = self.state

        if self.grid[UAV_row1, UAV_col1] > 0.0:
            self.visited1.add((UAV_row1, UAV_col1))  # mark visited cell

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol1 -= 1
            elif action == UP:
                nrow1 -= 1
            elif action == RIGHT:
                ncol1 += 1
            elif action == DOWN:
                nrow1 += 1
            elif action == STAY:
                pass
        else:  # invalid action, no change in UAV position
            mode = 'invalid'

        # new state
        self.state = (nrow1, ncol1, nmode)
        print(self.state)

    def get_reward(self):
        reward = 0
        UAV_row1, UAV_col1, mode = self.state
        nrows, ncols = self.grid.shape
        if UAV_row1 == UAV1_end[0] and UAV_col1 == UAV1_end[1]:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (UAV_row1, UAV_col1) in self.visited1:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.grid)
        nrows, ncols = self.grid.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the UAV
        row1, col1, valid = self.state
        canvas[row1, col1] = UAV_mark1
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        UAV_row1, UAV_col1, mode = self.state
        nrows, ncols = self.grid.shape
        if UAV_row1 == UAV1_end[0] and UAV_col1 == UAV1_end[1]:
            return 'win'
        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row1, col1, mode = self.state
        else:
            row1, col1 = cell
        actions = [0, 1, 2, 3, 4]
        nrows, ncols = self.grid.shape
        if row1 == 0:
            actions.remove(1)
        elif row1 == nrows - 1:
            actions.remove(3)

        if col1 == 0:
            actions.remove(0)
        elif col1 == ncols - 1:
            actions.remove(2)

        if row1 > 0 and self.grid[row1 - 1, col1] == 0.0:
            actions.remove(1)
        if row1 < nrows - 1 and self.grid[row1 + 1, col1] == 0.0:
            actions.remove(3)

        if col1 > 0 and self.grid[row1, col1 - 1] == 0.0:
            actions.remove(0)
        if col1 < ncols - 1 and self.grid[row1, col1 + 1] == 0.0:
            actions.remove(2)

        return actions

def show(U):
    plt.grid('on')
    nrows, ncols = U.grid.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(U.grid)
    for row, col in U.visited1:
        canvas[row, col] = 0.6
    for row, col in U.visited2:
        canvas[row, col] = 0.6
    UAV_row1, UAV_col1, _ = U.state
    canvas[UAV_row1, UAV_col1] = 0.3
    canvas[UAV1_end[0], UAV1_end[1]] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


uav = UAV(grid)
canvas, reward, game_over = uav.act(DOWN)
print("reward= ", reward)
show(uav)


def play_game(model, U, uav1):
    U.reset(uav1)
    envstate = U.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = U.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets



def completion_check(model, u):
    for cell in u.in_range_cells:
        if not u.valid_actions(cell):
            return False
        if not play_game(model, u, cell):
            return False
    return True

def qtrain(model, grid, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    Quav = UAV(grid)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    n_free_cells = len(Quav.in_range_cells)
    hsize = Quav.grid.size // 2  # history window size
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0
        uav1= random.choice(Quav.in_range_cells)
        Quav.reset(uav1)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = Quav.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = Quav.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = Quav.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9: epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, Quav):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)




def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model



model = build_model(grid)
qtrain(model,grid,n_epoch=1000,max_memory=8*grid.size,data_size=32)