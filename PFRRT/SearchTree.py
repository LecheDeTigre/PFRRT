import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal

figure, axes = plt.subplots( 1 )

class Node:
    def __init__(self, state, cost_to_node, cost_to_goal):
        self.state = state
        self.cost_to_node = cost_to_node
        self.cost_to_goal = cost_to_goal
        self.edges = []
        self.child_nodes = []
        self.parent_node = None
        self.parent_edge = None
    
    def addEdge(self, actuation, child_node):
        self.edges.append(actuation)
        self.child_nodes.append(child_node)

        child_node.parent_node = self
        child_node.cost_to_node += self.cost_to_node

        child_node.parent_edge = actuation

    # def __repr__(self):
    #     representation = "State: " + self.state + "\n"
    #     representation = representation + "Cost to Node: " + self.cost_to_node + "\n"
    #     representation = representation + "Cost to Goal: " + self.cost_to_goal + "\n"

    #     return representation
    
    # def __str__(self):
    #     representation = "State: " + self.state + "\n"
    #     representation = representation + "Cost to Node: " + self.cost_to_node + "\n"
    #     representation = representation + "Cost to Goal: " + self.cost_to_goal + "\n"

    #     return representation

class SearchTree:
    def __init__(self, root_node, N_rollouts, propagation_model, observation_model, path_reference, Q, R, dt, 
                 gamma, checkCollision, searchProblem):
        self.nodes = [root_node]
        self.edges = []

        self.best_node = root_node
        self.root_node = root_node
        
        self.feasible_trajectory_found = False

        self.N_rollouts = N_rollouts
        self.propagation_model = propagation_model
        self.observation_model = observation_model

        self.path_reference = path_reference

        self.Q = Q
        self.R = R

        self.Q_inv = np.linalg.inv(self.Q)
        self.R_inv = np.linalg.inv(self.R)

        self.dt = dt

        self.checkCollision = checkCollision
        
        self.stateCost = searchProblem.StateCost
        self.estimatedCostToGoal = searchProblem.EstimatedCostToGoal
        self.invalidState = searchProblem.InvalidState
        self.inGoalSet = searchProblem.InGoalSet

        self.gamma = gamma

    def getRootNode(self):
        return self.root_node
    
    def getBestNode(self):
        return self.best_node
    
    def getRandomNode(self):
        return random.choice(self.nodes)
    
    def pullBackBestNode(self):
        self.best_node = self.best_node.parent_node

    def addBranch(self, tree_node, nodes, actuations):
        self.nodes.extend(nodes)
        self.edges.extend(actuations)

        tree_node.addEdge(actuations[0], nodes[0])

        for node in nodes:
            # import pdb; pdb.set_trace()
            # if (self.best_node.cost_to_node + self.best_node.cost_to_goal > node.cost_to_node + node.cost_to_goal):
            if (self.best_node.cost_to_goal > node.cost_to_goal):
                self.best_node = node

                print(self.best_node.state)

    def sampleRandomState(self, prev_state, desired_observation):

        random_actuation = self.propagation_model.sampleRandomInput(prev_state=prev_state, dt = self.dt)
        feed_forward_state = self.propagation_model.step(prev_state, random_actuation, self.dt)

        pt_ref = self.path_reference.getClosestPoint(feed_forward_state)

        H = self.observation_model.getHMatrix(feed_forward_state, pt_ref)
        expected_observation = self.observation_model.getObservation(feed_forward_state, pt_ref)

        mat_1 = np.matmul(self.Q, np.transpose(H))
        mat_2 = np.matmul(H, mat_1)
        mat_3 = mat_2 + self.R

        L = np.matmul(mat_1, np.linalg.inv(mat_3))

        Sigma = np.linalg.inv(np.matmul(np.transpose(H), np.matmul(self.R_inv, H)) + self.Q_inv)
        next_state = feed_forward_state + np.matmul(L, desired_observation-expected_observation)

        sampled_state = np.random.multivariate_normal(next_state, Sigma, 1)[0]
        desired_future_observation = desired_observation

        diff = desired_future_observation-expected_observation

        norm = np.matmul(diff, np.matmul(np.linalg.inv(mat_3), np.transpose(diff)))

        candidate_likelihood = np.exp(-0.5*norm) # Needs to be a lambda

        future_observation_likelihood = candidate_likelihood # multivariate_normal.pdf(desired_future_observation, mean=expected_observation, cov=mat_3)

        # import pdb; pdb.set_trace()

        # print(feed_forward_state)
        # print(expected_observation)
        # print(norm)
        # print(future_observation_likelihood)

        return (sampled_state, random_actuation, future_observation_likelihood, expected_observation)

    def rolloutTrajectory(self, root_node, N_states, goal_state, desired_observation):
        rollout_states = np.zeros([self.N_rollouts, N_states, self.propagation_model.N_states])
        rollout_actuations = np.zeros([self.N_rollouts, N_states, self.propagation_model.N_actuations])

        rollout_weights = np.zeros([self.N_rollouts, N_states])

        optimal_states = np.zeros([N_states, self.propagation_model.N_states])
        optimal_actuations = np.zeros([N_states, self.propagation_model.N_actuations])

        success = True

        optimal_nodes = []

        for k in range(0, N_states):
            N_w = 0
            N_eff = 0
            sum_weights = 0.

            print("k: "+str(k))
            
            for i in range(0, self.N_rollouts):
                print("i: " + str(i))
                prev_state = None
                prev_weight = None

                if k == 0:
                    prev_state = root_node.state
                    prev_weight = 1/self.N_rollouts
                else:
                    prev_state = rollout_states[i][k-1]
                    prev_weight = rollout_weights[i][k-1]

                (sampled_state, random_actuation, future_observation_likelihood, expected_observation) = self.sampleRandomState(prev_state, desired_observation)

                rollout_states[i][k] = sampled_state
                rollout_actuations[i][k] = random_actuation

                invalid_state = self.invalidState(expected_observation)
                check_collision = self.checkCollision(sampled_state)

                if invalid_state or check_collision:
                    rollout_weights[i][k] = 0.

                    if invalid_state:
                        print("invalid state")
                    else:
                        print("check collision")
                else:
                    rollout_weights[i][k] = prev_weight*future_observation_likelihood

                print("state, invalid_state, check_collision, rollout_weight: " + str(rollout_states[i][k]) + ", " + str(invalid_state) + ", " + str(check_collision) + ", " + str(rollout_weights[i][k]))

                sum_weights += rollout_weights[i][k]

                N_eff += rollout_weights[i][k]*rollout_weights[i][k]

            N_w = sum_weights
            # print(N_w)

            if N_w == 0:
                print("Fail")
                # print(k)
                # import pdb; pdb.set_trace()
                if k ==0:
                    success = False
                else:
                    success = True

                rollout_states = rollout_states[:, 0:k]

                optimal_actuations = optimal_actuations[0:k]
                optimal_states = optimal_states[0:k]

                break

            for i in range(0, self.N_rollouts):
                rollout_weights[i][k] = rollout_weights[i][k]/sum_weights

                optimal_states[k] += rollout_weights[i][k]*rollout_states[i][k]
                optimal_actuations[k] += rollout_weights[i][k]*rollout_actuations[i][k]

            N_eff = 1/N_eff

            if N_eff <= self.gamma*self.N_rollouts:
                resampled_idx = np.random.choice(np.arange(0, self.N_rollouts), self.N_rollouts, p = rollout_weights[:,k])
                rollout_states[:, 0] = rollout_states[resampled_idx, 0]

            expected_observation = self.observation_model.getObservation(optimal_states[k], self.path_reference.getClosestPoint(optimal_states[k]))
            parent_node_observation = self.observation_model.getObservation(prev_state, self.path_reference.getClosestPoint(prev_state))

            optimal_state_cost = self.stateCost(expected_observation, desired_observation)

            estimated_cost_to_goal = None
            
            if self.inGoalSet(optimal_states[k]):
                estimated_cost_to_goal = 0
                self.feasible_trajectory_found = True
            else:
                estimated_cost_to_goal = self.estimatedCostToGoal(optimal_states[k], goal_state)

            optimal_nodes.append(Node(optimal_states[k], optimal_state_cost, estimated_cost_to_goal))
            
            print("optimal_state["+str(k)+"]: "+str(optimal_states[k]))
            print("optimal_state_costs: "+str((optimal_state_cost, estimated_cost_to_goal, optimal_state_cost+estimated_cost_to_goal)))

            if k != 0:
                curr_node = optimal_nodes[-1]
                prev_node = optimal_nodes[-2]
                
                # import pdb; pdb.set_trace()

                prev_node.addEdge(optimal_actuations[k], curr_node)

        print("root_node.state: "+str(root_node.state))

        circle_1 = plt.Circle((5., 0.), 5., fill=False)
        circle_2 = plt.Circle((5+4.9*np.cos(3*np.pi/4), 0+4.9*np.sin(3*np.pi/4)), 0.15, fill=True)

        axes.set_aspect(1)
        plt.xlim([-1., 6.])
        plt.ylim([-1., 6.])
        axes.add_artist(circle_1)

        idx = 0

        print("Display:")

        for rollout_state in rollout_states:
            axes.plot(rollout_state[:, 0], rollout_state[:, 1], label = str(idx))
            axes.plot(rollout_state[:, 0], rollout_state[:, 1], 'x')

            # print(rollout_state)

            # print("rollout_state:" +str(rollout_state))
            # print("rollout_weights[idx]"+str(rollout_weights[idx]))

            idx += 1

        axes.plot(optimal_states[:, 0], optimal_states[:, 1], 'r-')
        axes.plot(optimal_states[:, 0], optimal_states[:, 1], 'ro')

        print("Best Node: "+str((self.getBestNode().state)))
        print("Best Node Costs: "+str((self.getBestNode().cost_to_node, self.getBestNode().cost_to_goal,self.getBestNode().cost_to_node+self.getBestNode().cost_to_goal)))

        axes.plot(self.getBestNode().state[0], self.getBestNode().state[1], 'ko')

        plt.legend(loc="upper left")

        axes.add_artist(circle_2)

        plt.draw()
        plt.waitforbuttonpress()

        for optimal_node in optimal_nodes:

            optimal_state = optimal_node.state
            cost_to_node = optimal_node.cost_to_node
            cost_to_goal = optimal_node.cost_to_goal

            print("state: " + str(optimal_state))
            print("observation: " + str(self.observation_model.getObservation(optimal_state, self.path_reference.getClosestPoint(optimal_state))))
            print("cost_to_node: " + str(cost_to_node))
            print("cost_to_goal: " + str(cost_to_goal))

        return (success, optimal_nodes, optimal_actuations)
        
    def searchTrajectory(self, goal_state, N_states, N_iter, desired_observation):
        
        for i in range(0, N_iter):
            print("\niter: "+str(i))

            choice = np.random.choice([0, 1], p=[0.5, 0.5])

            print(choice)

            if self.feasible_trajectory_found:
                if choice == 0:
                    sampled_node = self.getRootNode()
                    print("Sampled Root Node")
                else:
                    sampled_node = self.getRandomNode()
                    print("Sampled Random Node")
            else:
                if choice == 0:
                    sampled_node = self.getBestNode()
                    print("Sampled Best Node")
                else:
                    sampled_node = self.getRandomNode()
                    print("Sampled Random Node")

            print("Sampled node: " + str(sampled_node.state))

            # for node in self.nodes:
            #     print(node.state)

            (success, rollout_nodes, rollout_actuations) = self.rolloutTrajectory(sampled_node, N_states, goal_state, desired_observation)
            
            if success:
                self.addBranch(sampled_node, rollout_nodes, rollout_actuations)
            elif not self.feasible_trajectory_found and choice == 0:
                self.pullBackBestNode()

            print("Best node after update: "+str((self.getBestNode().state)))
            print("Best node Costs after update: "+str((self.getBestNode().cost_to_node, self.getBestNode().cost_to_goal,self.getBestNode().cost_to_node+self.getBestNode().cost_to_goal)))


        return self.getBestNode()
        
