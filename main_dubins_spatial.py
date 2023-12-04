import matplotlib.pyplot as plt
import numpy as np

from ObservationModels import CurvilinearCoordinateSystemSpatialDubins
from PFRRT import SearchTree
from References import CirclePath
from Models import DubinsCarSpatial

def inGoalSet(state, goal_state):
    if(np.linalg.norm(state[0:2]-goal_state[0:2]) <= 0.25) and abs(state[2]-goal_state[2]) <= np.pi/12:
        return True
    else:
        return False

def stateCost(observation, desired_observation, parent_node_observation):
    # import pdb; pdb.set_trace()
    return np.matmul(observation-desired_observation, np.matmul(np.diag([5, 5/(np.pi/2)]), np.transpose(observation-desired_observation)))

def estimatedCostToGoal(optimal_state, goal_state, expected_observation, desired_observation):
    return 10*np.linalg.norm(optimal_state[0:2]-goal_state[0:2])

def checkCollision(state):
    centres = [[5+4.9*np.cos(3*np.pi/4), 0+4.9*np.sin(3*np.pi/4)]]
    radii = [0.2]

    for (centre, radius) in zip(centres, radii):
        # import pdb; pdb.set_trace()
        if (np.linalg.norm(centre-state[0][0:2]) <= radius):
            return True

    return False

def invalidState(observation):
    # import pdb; pdb.set_trace()
    y_lim = 0.2*1.2
    psi_lim = np.pi/12*1.2

    if (-y_lim <= observation[0] and observation[0] <= y_lim) and (-psi_lim <= observation[1] and observation[1] <= psi_lim):
        return False
    else:
        return True
    
initial_state = np.array([0., 0., np.pi/2])
goal_state = np.array([5., 5., 0.])

circle_path = CirclePath.CirclePath(np.array([5., 0.]), 5.0)
closest_pt_ref = circle_path.getClosestPoint(initial_state)

# print(closest_pt_ref)

(cross_track_error, heading_error) = CurvilinearCoordinateSystemSpatialDubins.getCurvilinearCoordinates(initial_state, closest_pt_ref)

# print(cross_track_error, heading_error)

N_rollouts = 100
N_states = 10

N_iter = 100

model = DubinsCarSpatial.DubinsCarSpatial()

Q = np.diag([0.05, 0.05, 0.0084])*1e-4
R = np.diag([0.2, 0.168])*1.25e-2

desired_observation = np.array([0., 0.])

ds = 0.1

gamma = 0.5

root_node = SearchTree.Node(initial_state, 0, 10*np.linalg.norm(initial_state[0:2]-goal_state[0:2]))

search_tree = SearchTree.SearchTree(root_node, N_rollouts, model, CurvilinearCoordinateSystemSpatialDubins, circle_path, Q, R, ds, stateCost, estimatedCostToGoal, invalidState, checkCollision, inGoalSet, gamma)

best_node = search_tree.searchTrajectory(goal_state, N_states, N_iter, desired_observation)

trajectory = []

curr_node = best_node

figure, axes = plt.subplots(2)

circle_1 = plt.Circle((5., 0.), 5., fill=False)
circle_2 = plt.Circle((5+4.9*np.cos(3*np.pi/4), 0+4.9*np.sin(3*np.pi/4)), 0.2, fill=True)

axes[0].set_aspect(1)
# plt.xlim([-1., 5.])
# plt.ylim([-7., 7.])
axes[0].add_artist(circle_1)
axes[0].add_artist(circle_2)

while curr_node != None:
    # print(curr_node.state)

    trajectory = [curr_node] + trajectory
    curr_node = curr_node.parent_node

optimal_states = np.array([node.state for node in trajectory])

# import pdb; pdb.set_trace()

optimal_actuations = np.array([node.parent_edge for node in trajectory][1:])

print(np.mean(optimal_actuations))

# print(optimal_states)

axes[0].plot(optimal_states[:, 0], optimal_states[:, 1])
axes[0].plot(optimal_states[:, 0], optimal_states[:, 1], 'gx')

axes[1].plot(optimal_actuations)

plt.show()