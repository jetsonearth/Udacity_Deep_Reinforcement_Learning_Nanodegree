import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA = 6, state_space = 500, gamma = 1.0, alpha = 0.05, epsilon = 0.001, eps_decay = 0.9999):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.state_space = state_space
        self.epsilon_min = 0.0001
        self.eps_decay = 0.9999
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
    def epsilon_policy(self, Q_state):
        
        """obtain the action probabilities corresponding to the epsilon-greedy policy"""
        policy_state = np.ones(self.nA) * self.epsilon /self.nA
        best_action = np.argmax(Q_state)
        policy_state[best_action] = 1 - self.epsilon + (self.epsilon / self.nA)
        
        return policy_state
        
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # obtain the action probabilities
        policy_state = self.epsilon_policy(self.Q[state])
        # choose the value of epsilon 
        self.epsilon = max(self.epsilon * self.eps_decay, self.epsilon_min)
        # perform an action
        action_state = np.random.choice(np.arange(self.nA), p = policy_state)
        
        return action_state

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        if not done:
            # choose next action using eps-greedy policy, with probabilities
            action = self.select_action(state)
            # obtain policy of next state
            policy_state = self.epsilon_policy(self.Q[next_state])
            # calculate expected value
            expected_value = np.dot(self.Q[next_state], policy_state)
            # update Q
            self.Q[state][action] += self.alpha*(reward+(self.gamma*expected_value)-self.Q[state][action])
        if done:
            # since done = 0, so next state is unavailable, therefore gamma is 0
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])