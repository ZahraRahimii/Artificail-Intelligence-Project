# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from operator import ne
import re
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    "*** YOUR CODE HERE ***"

    visited = []
    stack = util.Stack() 
    stack.push((problem.getStartState(), []))
    
    while not stack.isEmpty():
        (next_node, path) = stack.pop()
        # print(next_node, path)
        if not next_node in visited:
            if problem.isGoalState(next_node):
                return path
            visited.append(next_node)
            neighbers = problem.getSuccessors(next_node)
            for neighber in neighbers:
                stack.push(( neighber[0], path + [neighber[1]])) # neighber[0] is the state and neighber[1] is the action
    
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    visited = []
    queue = util.Queue() 
    queue.push((problem.getStartState(), []))
    

    while not queue.isEmpty():
        (next_node, path) = queue.pop()
        # print("next node breadth:\n", next_node)

        if not next_node in visited:
            if problem.isGoalState(next_node):
                return path
            
            visited.append(next_node)
            neighbers = problem.getSuccessors(next_node)
            # print("neighbers:\n", neighbers)
            for neighber in neighbers:
                queue.push(( neighber[0], path + [neighber[1]])) # neighber[0] is the state and neighber[1] is the action


    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pQueue = util.PriorityQueue()
    visited = set()

    pQueue.push((problem.getStartState(), []), 0)

    while not pQueue.isEmpty():
        (next_node, path)  = pQueue.pop()
        # print(next_node, path)
        if not next_node in visited:
            if problem.isGoalState(next_node):
                return path
            visited.add(next_node)
            neighbers = problem.getSuccessors(next_node)
            for neighber in neighbers:
                pQueue.push((neighber[0], path + [neighber[1]]), problem.getCostOfActions(path + [neighber[1]])) # neighber[0] is the state and neighber[1] is the action

    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pHQueue = util.PriorityQueue()
    visited = []

    pHQueue.push((problem.getStartState(), []), 0)

    while not pHQueue.isEmpty():
        (next_node, path) = pHQueue.pop()
        # print(next_node, path)
        if not next_node in visited:
            if problem.isGoalState(next_node):
                return path
            visited.append(next_node)
            neighbers = problem.getSuccessors(next_node)
            for neighber in neighbers:
                pHQueue.push((neighber[0], path + [neighber[1]]), problem.getCostOfActions(path + [neighber[1]]) + heuristic(neighber[0], problem)) # neighber[0] is the state and neighber[1] is the action
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
