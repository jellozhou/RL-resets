import math
import random
import collections

# source: partially from https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial1/qlearn.py
class QLearn:
    """Q-Learning class. Implements the Q-Learning algorithm."""

    def __init__(self,
                 actions,
                 epsilon, # exploration vs exploitation
                 learning_rate, # alpha
                 gamma):
        """Initialize an empty dictionary for Q-Values."""
        # Q-Values are stored in a dictionary, with the state-action
        self.q = {}
        # print(self.q)

        # Epsilon is the exploration factor. A higher epsilon
        # encourages more exploration, risking more but potentially
        # gaining more too.
        self.epsilon = epsilon

        # Alpha is the learning rate. If Alpha is high, then the
        # learning is faster but may not converge. If Alpha is low,
        # the learning is slower but convergence may be more stable.
        self.learning_rate = learning_rate

        # Gamma is the discount factor.
        # It prioritizes present rewards over future ones.
        self.gamma = gamma

        # Actions available in the environment
        self.actions = actions

        # history of stability
        self.prev_max_q_indices = None # previous Q table

    def getQ(self, state, action):
        """Get Q value for a state-action pair.

        If the state-action pair is not found in the dictionary,
            return 0.0 if not found in our dictionary
        """
        return self.q.get((state, action), 0.0) # returns the dict entry

    def learnQ(self, state, action, reward, target):
        # takes in target value
        """Updates the Q-value for a state-action pair.

        The core Q-Learning update rule.
            Q(s, a) += alpha * (reward(s,a) + max(Q(s')) - Q(s,a))

        This function updates the Q-value for a state-action pair
        based on the reward and maximum estimated future reward.
        """
        oldv = self.q.get((state, action), None)
        if oldv is None:
            # If no previous Q-Value exists, then initialize
            # it with the current reward
            self.q[(state, action)] = reward
        else:
            # Update the Q-Value with the weighted sum of old
            # value and the newly found value.
            #
            # Alpha determines how much importance we give to the
            # new value compared to the old value.
            self.q[(state, action)] = oldv + self.learning_rate * (target - oldv)


    def QStable(self):
        """Check if the relative maximum Q-value indices for all states are the same as the previous episode."""
        if self.prev_max_q_indices is None:
            return False  # No previous max indices to compare to in the first episode
        
        current_max_q_indices = {}
        
        # Get the index of the maximum Q-value for each state
        for state in self.q.keys():
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q_index = q_values.index(max(q_values))  # Find the action with the max Q-value
            current_max_q_indices[state] = max_q_index

        # Compare the current max indices with the previous ones
        for state, max_index in current_max_q_indices.items():
            if self.prev_max_q_indices.get(state) != max_index:
                return False  # The relative max Q-value index has changed for this state

        # If all states have the same relative max Q-value index, we consider the Q-table stable
        return True

    # def QStableForN(self, n):
    #     """Check if QStable has returned True for the past n episodes."""
    #     return len(self.stability_history) == n and all(self.stability_history)

    # def updateStability(self):
    #     """Update stability history after each episode."""
    #     self.stability_history.append(self.QStable())

    def chooseAction(self, state):
        """Epsilon-Greedy approach for action selection."""
        # there could be annealing for epsilon, but not necessary
        if random.random() < self.epsilon:
            # With probability epsilon, we select a random action
            action = random.choice(self.actions)
        else:
            # With probability 1-epsilon, we select the action
            # with the highest Q-value
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            # If there are multiple actions with the same Q-Value,
            # then choose randomly among them
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    # function to call that updates q values from trajectory
    def learn(self, state1, action1, reward, state2):
        """Get the maximum Q-Value for the next state."""
        maxqnew = max([self.getQ(state2, a) for a in self.actions])

        # Learn the Q-Value based on current reward and future
        # expected rewards.
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)

    def save_max_q_indices(self):
        """Save the relative maximum Q-value indices for each state."""
        self.prev_max_q_indices = {}
        
        # Store the relative maximum Q-value indices for each state
        for state in self.q.keys():
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q_index = q_values.index(max(q_values))
            self.prev_max_q_indices[state] = max_q_index