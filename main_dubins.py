import matplotlib.pyplot as plt
import numpy as np

from ObservationModels import CurvilinearCoordinateSystemDubins
from PFRRT import SearchTree
from References import CirclePath
from Models import DubinsCar

def inGoalSet(state, goal_state):
    if(np.linalg.norm(state[0:2]-goal_state[0:2]) <= 0.25) and abs(state[2]-goal_state[2]) <= np.pi/12:
        return True
    else:
        return False

def stateCost(observation, desired_observation, parent_node_observation):
    return np.matmul(observation-desired_observation, np.matmul(np.diag([5, 2, 1]), np.transpose(observation-desired_observation)))

def estimatedCostToGoal(optimal_state, goal_state, expected_observation, desired_observation):
    optimal_state_projection = circle_path.getClosestPoint(optimal_state)
    goal_state_projection = circle_path.getClosestPoint(goal_state)

    x, y, psi, s = optimal_state_projection[0], optimal_state_projection[1], optimal_state_projection[2], optimal_state_projection[3]
    x_ref, y_ref, psi_ref, s_ref = goal_state_projection[0], goal_state_projection[1], goal_state_projection[2], goal_state_projection[3]

    eY_ref = -(goal_state[0]-x_ref)*np.sin(psi_ref) + (goal_state[1]-y_ref)*np.cos(psi_ref)
    eY = -(optimal_state[0]-x)*np.sin(psi) + (optimal_state[1]-y)*np.cos(psi)
    
    return 1.2* ((eY-eY_ref)**2 + (s-s_ref)**2)

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
    y_lim = 0.20
    psi_lim = np.pi/3

    if (-y_lim <= observation[1] and observation[1] <= y_lim):
        return False
    else:
        return True
    
initial_state = np.array([0., 0., np.pi/2, 0., 0.])
goal_state = np.array([5., 5., 0., 0., 0.])

circle_path = CirclePath.CirclePath(np.array([5., 0.]), 5.0)
closest_pt_ref = circle_path.getClosestPoint(initial_state)

# print(closest_pt_ref)

(velocity, cross_track_error, dist) = CurvilinearCoordinateSystemDubins.getCurvilinearCoordinates(initial_state, closest_pt_ref)

# print(cross_track_error, heading_error)

N_rollouts = 100
N_states = 10

N_iter = 100

model = DubinsCar.DubinsCar()

Q = np.diag([0.05, 0.05, 0.0084, 1e-2, 0.0084])*1e-4
R = np.diag([1.0, 1e-4, 0.1])

desired_observation = np.array([0., 0., 0.])

ds = 0.1

gamma = 0.5

pt_ref = circle_path.getClosestPoint(initial_state)

print(pt_ref)

print(CurvilinearCoordinateSystemDubins.getCurvilinearCoordinates(initial_state, pt_ref))
print(desired_observation)

initial_state_cost = stateCost(CurvilinearCoordinateSystemDubins.getCurvilinearCoordinates(initial_state, circle_path.getClosestPoint(initial_state)), desired_observation, CurvilinearCoordinateSystemDubins.getCurvilinearCoordinates(initial_state, circle_path.getClosestPoint(initial_state)))

print(initial_state_cost)

root_node = SearchTree.Node(initial_state, initial_state_cost, estimatedCostToGoal(initial_state, goal_state, None, None))

search_tree = SearchTree.SearchTree(root_node, N_rollouts, model, CurvilinearCoordinateSystemDubins, circle_path, Q, R, ds, stateCost, estimatedCostToGoal, invalidState, checkCollision, inGoalSet, gamma)

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

print(optimal_states)
print(optimal_states[:, 0])
print(optimal_states[:, 1])

plt.xlim([-1., 6.])
plt.ylim([-1., 6.])

axes[0].plot(optimal_states[:, 0], optimal_states[:, 1])
axes[0].plot(optimal_states[:, 0], optimal_states[:, 1], 'gx')

axes[1].plot(optimal_actuations)

axes[0].set_aspect(1)
axes[1].set_aspect(1)

plt.show()