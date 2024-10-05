import numpy as np
import random
import matplotlib.pyplot as plt

# 0 represents free spaces, 1 represents walls.
maze = np.array([
    [1]*30,
    [1]*30,
    [1]*30,
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1] * 30,
    [1] * 30,
    [1] * 30
    ])


start = (14, 3)
finish = (15, 25)

# Initialize Q-table
state_size = maze.size
action_size = 4  # Up, Down, Left, Right
Q = np.zeros((state_size, action_size))

# Define RL parameters
alpha = 0.1  # Learning coefficient

gamma = 0.9  # Discount factor
# this is a parameter that determines how much the algorithm values future rewards
# compared to immediate rewards.

epsilon = 1.0  # Exploration rate

epsilon_decay = 0.995
min_epsilon = 0.01

episodes = 100000  # This value can be adjusted based on the complexity of the algorithm. The algorithm may not


# necessarily use all episodes.


def state_to_index(state):
    return state[0] * maze.shape[1] + state[1]


def get_valid_actions(position):
    x, y = position
    actions = []
    if x > 0 and maze[x - 1, y] == 0: actions.append(0)  # Up
    if x < maze.shape[0] - 1 and maze[x + 1, y] == 0: actions.append(1)  # Down
    if y > 0 and maze[x, y - 1] == 0: actions.append(2)  # Left
    if y < maze.shape[1] - 1 and maze[x, y + 1] == 0: actions.append(3)  # Right
    return actions


def move(position, action):
    y: object
    x, y = position
    if action == 0: return x - 1, y  # Up
    if action == 1: return x + 1, y  # Down
    if action == 2: return x, y - 1  # Left
    if action == 3: return x, y + 1  # Right


def train_q_learning():
    stop = False
    allPaths = []
    global epsilon
    for episode in range(episodes):
        state = start
        path = [state]
        while state != finish:
            state_idx = state_to_index(state)
            valid_actions = get_valid_actions(state)

            if not valid_actions:
                print("No valid actions found")
                break

            if random.uniform(0, 1) < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(Q[state_idx])

            next_state = move(state, action)

            # Check if next_state is within bounds and not a wall
            if next_state == finish:
                reward = 100
                done = True
            elif maze.shape[0] > next_state[0] >= 0 == maze[
                next_state[0], next_state[1]] and 0 <= next_state[1] < maze.shape[1]:
                reward = -1
                done = False
            else:
                next_state = state  # Stay in the same state if move is invalid
                reward = -10  # Penalty
                done = False

            next_state_idx = state_to_index(next_state)
            best_next_action = np.argmax(Q[next_state_idx])
            Q[state_idx, action] += alpha * (
                    reward + gamma * Q[next_state_idx, best_next_action] - Q[state_idx, action])

            state = next_state
            path.append(state)

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        """
        If the path to be added already exists in the list and appears more than once,
        we can conclude that we have found the optimal path.
        """
        if len(allPaths) >= 2:
            if path == allPaths[-1] and path == allPaths[-2]:
                break
        allPaths.append(path)

    return allPaths


def show_solution(path):
    maze_with_path = np.copy(maze)
    for (x, y) in path:
        maze_with_path[x, y] = 0.5  # To have path with a different value

    plt.figure()
    plt.imshow(maze_with_path, cmap='viridis', origin='upper')
    plt.scatter(start[1], start[0], c='green', label='S', edgecolor='black')
    plt.scatter(finish[1], finish[0], c='red', label='F', edgecolor='black')

    # Draw arrows along the path
    for j in range(len(path) - 1):
        start_point = path[j]
        end_point = path[j + 1]
        plt.annotate('', xy=end_point[::-1], xytext=start_point[::-1],
                     arrowprops=dict(facecolor='white', shrink=0.05))

    plt.legend()
    plt.title('Maze w/ Path')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.show()


def visualize_heat_map(Q):
    Q_heat = Q
    action_to_visualize = 3

    # Q-values matrix for actions
    q_values_map = np.full(maze.shape, np.nan)

    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y] == 0:  # To consider only valid states (paths)
                state_idx = x * maze.shape[1] + y
                q_values_map[x, y] = Q_heat[state_idx, action_to_visualize]

    plt.figure(figsize=(10, 8))
    plt.imshow(q_values_map, cmap='viridis', origin='upper', interpolation='nearest')
    plt.colorbar(label='Q-value')
    plt.title(f'Q-values Heat Map')
    plt.scatter(start[1], start[0], c='green', label='Start', edgecolor='black')
    plt.scatter(finish[1], finish[0], c='red', label='Finish', edgecolor='black')
    plt.legend()
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def getSolution():
    all_paths = train_q_learning()
    final_path = all_paths[-1]  # Get the path from the last episode

    #show_solution(final_path)
    #visualize_heat_map(Q)

    return final_path, maze.shape

getSolution()