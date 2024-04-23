import matplotlib.pyplot as plt
import numpy as np

from ObservationModels.CircularObstacle import CircularObstacle
from ObservationModels.CurvilinearCoordinateSystemDubins import CurvilinearCoordinateSystemDubins
from SearchProblems.SimpleCorridorTrackingProblem import SimpleCorridorTrackingProblem

from Models import DubinsCar
from PFRRT import SearchTree
from References.CirclePath import CirclePath

def checkCollision(state, obstacles):
    for obstacle in obstacles:
        # import pdb; pdb.set_trace()
        if obstacle.checkCollision(state):
            return True
        
def getObstacles():
    obstacles = []
    
    obstacles.append(CircularObstacle([5+4.9*np.cos(3*np.pi/4), 0+4.9*np.sin(3*np.pi/4)], 0.15))
    #obstacles.append(CircularObstacle([5+4.9*np.cos(3*np.pi/4), 0+4.9*np.sin(3*np.pi/4)], 0.15))
    
    return obstacles
    
def ObstacleProximityCost(state, obstacles):
    cost = 1.0
    weight = 0.5
    for obstacle in obstacles:
        cost *= np.exp(-weight*obstacle.getObservation(state, None)**2)
    
    return cost
    
initial_state = np.array([0., 0., np.pi/2, 0., 0.])
goal_state = np.array([5., 5., 0., 0., 0.])

circle_path = CirclePath(np.array([5., 0.]), 5.0)
closest_pt_ref = circle_path.getClosestPoint(initial_state)

# print(closest_pt_ref)
obstacles = getObstacles()

check_collision_lambda = lambda state: checkCollision(state, obstacles=obstacles)

curvilinear_coordinate_system_dubins = CurvilinearCoordinateSystemDubins()

simple_corridor_tracking_problem = SimpleCorridorTrackingProblem(goal_state=goal_state, GetClosestPoint=circle_path.getClosestPoint)

(velocity, cross_track_error, dist) = curvilinear_coordinate_system_dubins.getObservation(initial_state, closest_pt_ref)

# print(cross_track_error, heading_error)

N_rollouts = 100
N_states = 10

N_iter = 50

model = DubinsCar.DubinsCar()

Q = np.diag([0.05, 0.05, 0.0084, 1e-2, 0.0084])*1e-4
R = np.diag([1.0, 1e-4, 0.1])

desired_observation = np.array([0., 0., 5*np.pi/2])

dt = 0.1

gamma = 0.5

pt_ref = circle_path.getClosestPoint(initial_state)

print(pt_ref)

print(curvilinear_coordinate_system_dubins.getObservation(initial_state, pt_ref))
print(desired_observation)

initial_state_cost = simple_corridor_tracking_problem.StateCost(curvilinear_coordinate_system_dubins.getObservation(initial_state, circle_path.getClosestPoint(initial_state)), desired_observation)

print(initial_state_cost)

root_node = SearchTree.Node(initial_state, initial_state_cost, simple_corridor_tracking_problem.EstimatedCostToGoal(initial_state, goal_state))

search_tree = SearchTree.SearchTree(root_node=root_node, N_rollouts=N_rollouts, propagation_model=model, observation_model=curvilinear_coordinate_system_dubins,
                                    path_reference=circle_path, Q=Q, R=R, dt=dt, gamma=gamma, checkCollision=check_collision_lambda, searchProblem=simple_corridor_tracking_problem)

best_node = search_tree.searchTrajectory(goal_state, N_states, N_iter, desired_observation)

trajectory = []

curr_node = best_node

figure, axes = plt.subplots(2)

circle_1 = plt.Circle((5., 0.), 5., fill=False)
circle_2 = plt.Circle((5+4.9*np.cos(3*np.pi/4), 0+4.9*np.sin(3*np.pi/4)), 0.15, fill=True)

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