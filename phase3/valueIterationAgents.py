# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from turtle import st
import mdp, util, math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

        states = mdp.getStates()
        actions = mdp.getPossibleActions(states[1])
        # print(actions)
        # print(mdp.getPossibleActions(states[1]))
        # print(mdp.getTransitionStatesAndProbs(states[2], actions[1]))
        # print(mdp.getReward(states[1], actions[1], states[2]))
        # print(mdp.isTerminal(states[1]))

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        for iteration in range(self.iterations):
            tmp_values = util.Counter()
            
            for state in mdp.getStates():
                max = -math.inf
                actions = mdp.getPossibleActions(state)
                for action in actions:
                    sum = 0
                    st_acs = mdp.getTransitionStatesAndProbs(state, action)
                    for st_ac in st_acs:
                        next_st = st_ac[0]
                        prob = st_ac[1]
                        R = mdp.getReward(state, action, next_st) 
                        sum += prob * (R + self.discount * self.values[next_st])
                    if sum > max:
                        max = sum
                if max != -math.inf:
                    tmp_values[state] = max
            for state in mdp.getStates():
                self.values[state] = tmp_values[state]
 
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        sum = 0
        mdp = self.mdp
        st_acs = mdp.getTransitionStatesAndProbs(state, action)
        gamma = self.discount
        for st_ac in st_acs:
            next_st = st_ac[0]
            prob = st_ac[1]
            R = mdp.getReward(state, action, next_st)
            sum += prob * ( R + gamma * self.values[next_st])
        
        return sum
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)
        value_action = []
        best_act = None
        best_val = -math.inf
        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_val:
                best_val = q_value
                best_act = action
            # (q_value, action) = max(value_action)
        # else:
        #     return None
        return best_act
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        counter = 0
        tmp_values = util.Counter()
        for iteration in range(self.iterations):
            if counter >= len(mdp.getStates()):
                counter = 0
            
            state = mdp.getStates()[counter]
            counter += 1
            if state == "TERMINAL_STATE":
                continue
            max = -math.inf
            actions = mdp.getPossibleActions(state)
            for action in actions:
                sum = 0
                st_acs = mdp.getTransitionStatesAndProbs(state, action)
                for st_ac in st_acs:
                    next_st = st_ac[0]
                    prob = st_ac[1]
                    R = mdp.getReward(state, action, next_st) 
                    sum += prob * (R + self.discount * self.values[next_st])
                if sum > max:
                    max = sum
            if max != -math.inf:
                tmp_values[state] = max
            
            # for state in mdp.getStates():
            self.values[state] = tmp_values[state]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"    
        mdp = self.mdp
        predecessores = {}
        queue = util.PriorityQueue()

        for state in mdp.getStates():
            if state != "TERMINAL_STATE":  
                actions = mdp.getPossibleActions(state)
                for action in actions:
                    st_acs = mdp.getTransitionStatesAndProbs(state, action)
                    for st_ac in st_acs:
                        next_st = st_ac[0]
                        prob = st_ac[1]
                        if next_st not in predecessores:
                            predecessores[next_st] = set()
                        predecessores[next_st].add(state)
        

        for state in mdp.getStates():
            if not mdp.isTerminal(state):
                max_val = -math.inf
                actions = mdp.getPossibleActions(state)
                for action in actions:
                    max_val = max(max_val, self.getQValue(state, action))              
                diff = math.fabs(self.values[state] - max_val)
                queue.push(state, -diff)
        
        for itr in range(self.iterations):
            if not queue.isEmpty():
                s = queue.pop()
                if mdp.isTerminal(s):
                    continue
                max_val = -math.inf
                actions = mdp.getPossibleActions(s)
                for action in actions:
                    max_val = max(max_val, self.getQValue(s, action)) 
                if max_val != -math.inf:
                    self.values[s] = max_val
                for p in predecessores[s]:
                    max_val = -math.inf
                    actions = mdp.getPossibleActions(p)
                    for action in actions:
                        max_val = max(max_val, self.getQValue(p, action)) 
                    diff = math.fabs(self.values[p] - max_val)
                    if diff > self.theta:
                        queue.update(p, -diff)
            
