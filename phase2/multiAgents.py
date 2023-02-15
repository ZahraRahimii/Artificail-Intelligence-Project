# multiAgents.py
# --------------
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


from zmq import curve_public
from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # print("legal moves", legalMoves)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        newGhostsPos = successorGameState.getGhostPositions()
        currGhostsPos = currentGameState.getGhostPositions()
        currFood = currentGameState.getFood()

        if successorGameState.isWin():
            return 100000
            
        score = successorGameState.getScore() - currentGameState.getScore()
        if action == Directions.STOP:
            score -= 20

        if len(newFood.asList()) < len(currFood.asList()):
            score += 200
        
        score -= len(newFood.asList()) * 20

      
        if newPos in currentGameState.getCapsules():
            score += 300 * len(successorGameState.getCapsules())

        foodDistance = []
        infinite = math.inf
        # neg_infinite = -math.inf
        # minFlag = 0
        # maxFlag = 0

        for food in newFood.asList():
            dis = manhattanDistance(newPos, food)
            if dis > 0:
                foodDistance.append(((dis), food))
            else:
                foodDistance.append(((0.5), food))

        
        minFoodDis, minFood = min(foodDistance)
        if minFoodDis < infinite:
            minFlag = 1

        # if minFlag:
        #     foodDistance = []
        #     for food in newFood.asList():
        #         dis = manhattanDistance(minFood, food)
        #         if dis > 0:
        #             foodDistance.append(dis)
        #         else:
        #             foodDistance.append(((0.0005), food))
        #     maxFoodDis = max(foodDistance)
        #     if maxFoodDis > neg_infinite:
        #         maxFlag = 1


        
        ghostDistance = []
        for g in newGhostsPos:
            dis = manhattanDistance(newPos, g)
            # print(dis)
            # if dis > 0.0:
            ghostDistance.append(dis)

        currGhostsDistance = []
        for g in currGhostsPos:
            dis = manhattanDistance(newPos, g)
            currGhostsDistance.append(dis)
        
        newMinGhostDis = min(ghostDistance)
        currMinGhostDis = min(currGhostsDistance)
        

        if sum(newScaredTimes):
            if currMinGhostDis < newMinGhostDis:
                score += 400
            else:
                score -= 200
        else:
            if currMinGhostDis < newMinGhostDis:
                score -= 200
            else:
                score += 400
    
        if minFlag:
             score += 100.0 / minFoodDis

        

        return score
        # print("new pos\n", newPos)
        # print("new food\n", newFood)
        # print("new Ghost states\n", newGhostStates)
        # print("new Scared times\n", newScaredTimes)

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        curr_score = -math.inf
        bestAction = ''
        agent_index = 1
        actions_successors = [(action, gameState.generateSuccessor(0, action)) for action in actions]
        for action_successor in actions_successors:
            # next_state = gameState.generateSuccessor(0, action)
            score = MinimaxAgent.min_agent(self, action_successor[1], 0, agent_index)
            if score > curr_score:
                bestAction = action_successor[0]
                curr_score = score
        return bestAction

        # util.raiseNotDefined()

    def min_agent(self, gameState, depth, agent_index):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        min_val = math.inf
        agent_actions = gameState.getLegalActions(agent_index)
        actions_successors = [(gameState.generateSuccessor(agent_index, action)) for action in agent_actions]
        
        for successor in actions_successors:
            if agent_index == range(gameState.getNumAgents())[-1]:
                min_val = min(min_val, MinimaxAgent.max_agent(self, successor, depth + 1))
            else:
                min_val = min(min_val, MinimaxAgent.min_agent(self, successor, depth, agent_index + 1))
        
        return min_val

    def max_agent(self, gameState, depth):
        # curr_depth = depth + 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        max_val = -math.inf
        agent_actions = gameState.getLegalActions(0)
        actions_successors = [(gameState.generateSuccessor(0, action)) for action in agent_actions]

        for successor in actions_successors:
            max_val = max(max_val, MinimaxAgent.min_agent(self, successor, depth, 1))
        
        return max_val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        best_score = -math.inf
        bestAction = ''
        agent_index = 1
        alpha = -math.inf
        beta =  math.inf
        actions_successors = [(action, gameState.generateSuccessor(0, action)) for action in actions]
        for action_successor in actions_successors:
            score = AlphaBetaAgent.min_agent(self, action_successor[1], 0, agent_index, alpha, beta)
            if score > best_score:
                bestAction = action_successor[0]
                best_score = score
            if score > beta:
                return bestAction
            alpha = max(alpha, score)
        return bestAction

    def min_agent(self, gameState, depth, agent_index, alpha1, beta1):
        if gameState.isWin() or gameState.isLose():  
            return self.evaluationFunction(gameState)
        
        min_val = math.inf
        actions = gameState.getLegalActions(agent_index)
        beta = beta1
        alpha = alpha1
        for action in actions:
            successor= gameState.generateSuccessor(agent_index,action)
            if agent_index == (range(gameState.getNumAgents())[-1]):
                min_val = min(min_val, AlphaBetaAgent.max_agent(self, successor, depth + 1, alpha, beta))
                if min_val < alpha:
                    return min_val
                beta = min(beta, min_val)
            else:
                min_val = min(min_val, AlphaBetaAgent.min_agent(self, successor, depth, agent_index+1, alpha, beta))
                if min_val < alpha:
                    return min_val
                beta = min(beta, min_val)
        return min_val
        

    def max_agent(self, gameState, depth, alpha1, beta1):
        if gameState.isWin() or gameState.isLose() or depth==self.depth:   
            return self.evaluationFunction(gameState)
        max_val = -math.inf
        actions = gameState.getLegalActions(0)
        alpha = alpha1
        beta = beta1
        for action in actions:
            successor= gameState.generateSuccessor(0, action)
            max_val = max (max_val, AlphaBetaAgent.min_agent(self, successor, depth, 1, alpha, beta))
            if max_val > beta:
                return max_val
            alpha = max(alpha, max_val)
        return max_val
        



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        best_score = -math.inf
        bestAction = ' '
        agent_index = 1
        actions_successors = [(action, gameState.generateSuccessor(0, action)) for action in actions]
        for action_successor in actions_successors:
            score = ExpectimaxAgent.exp_agent(self, action_successor[1], 0, agent_index)
            if score > best_score:
                bestAction = action_successor[0]
                best_score = score
           
        return bestAction

    def exp_agent(self, gameState, depth, agent_index): 
        if gameState.isWin() or gameState.isLose():  
            return self.evaluationFunction(gameState)
        
        total_val = 0
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index,action)
            if agent_index == (range(gameState.getNumAgents())[-1]):
                exp_val = ExpectimaxAgent.max_agent(self, successor, depth + 1)
                
            else:
                exp_val = ExpectimaxAgent.exp_agent(self, successor, depth, agent_index+1)
            total_val += exp_val

        actions_num = len(actions)
        if not actions_num:
            return 0
        return float(total_val)/float(actions_num)
        

    def max_agent(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth==self.depth:   
            return self.evaluationFunction(gameState)
        max_val = -math.inf
        actions = gameState.getLegalActions(0)
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            max_val = max (max_val, ExpectimaxAgent.exp_agent(self, successor, depth, 1))
            
        return max_val




def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
        # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # currPos = currentGameState.getPacmanPosition()
    # newFood = currentGameState.getFood()
    # newGhostStates = currentGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    
    # currGhostsPos = currentGameState.getGhostPositions()
    # currFood = currentGameState.getFood()

    # if currentGameState.isWin():
    #     return 100000
        
    # score = currentGameState.getScore() - currentGameState.getScore()
    # # if action == Directions.STOP:
    # #     score -= 20

    # if len(newFood.asList()) < len(currFood.asList()):
    #     score += 200
    
    # score -= len(newFood.asList()) * 20

    
    # if currPos in currentGameState.getCapsules():
    #     score += 300 * len(currentGameState.getCapsules())

    # foodDistance = []
    # infinite = math.inf
    # neg_infinite = -math.inf
    # # minFlag = 0
    # # maxFlag = 0

    # for food in newFood.asList():
    #     dis = manhattanDistance(currPos, food)
    #     if dis > 0:
    #         foodDistance.append(((dis), food))
    #     else:
    #         foodDistance.append(((0.5), food))

    
    # minFoodDis, minFood = min(foodDistance)
    # if minFoodDis < infinite:
    #     minFlag = 1
    
    # if minFlag:
    #     foodDistance = []
    #     for food in newFood.asList():
    #         dis = manhattanDistance(minFood, food)
    #         if dis > 0:
    #             foodDistance.append(dis)
    #         else:
    #             foodDistance.append(((0.0005), food))
    #     maxFoodDis = max(foodDistance)
    #     if maxFoodDis > neg_infinite:
    #         maxFlag = 1


    
    # ghostDistance = []
    # for g in currGhostsPos:
    #     dis = manhattanDistance(currPos, g)
    #     # print(dis)
    #     # if dis > 0.0:
    #     ghostDistance.append(dis)

    # currGhostsDistance = []
    # for g in currGhostsPos:
    #     dis = manhattanDistance(currPos, g)
    #     currGhostsDistance.append(dis)
    
    # newMinGhostDis = min(ghostDistance)
    # currMinGhostDis = min(currGhostsDistance)
    

    # if sum(newScaredTimes):
    #     if currMinGhostDis < newMinGhostDis:
    #         score += 400
    #     else:
    #         score -= 200
    # else:
    #     if currMinGhostDis < newMinGhostDis:
    #         score -= 200
    #     else:
    #         score += 400

    # if minFlag:
    #         score += 100.0 / minFoodDis

    # return score

    pacman_pos = currentGameState.getPacmanPositoin()
    ghostsStates = currentGameState.getGhostStates()
    foods = currentGameState.getFood()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostsStates]

    if currentGameState.isWin():
        return 10000
    
    ghostDis = []
    for ghost in ghostsStates:
        ghostDis.append(manhattanDistance(pacman_pos, ghost.getPostion()))
    
    foodDis = []
    for food in foods.asList():
        foodDis.append(manhattanDistance(pacman_pos, food))
    
    capsules = len(currentGameState.getCapsules())
    minFoodDis = min(foodDis)
    notEaten = sum(foods.asList(False))
    scaredSum = sum(scaredTimes)
    minGhostDis = sum(ghostDis)
    ghostDisSum = sum(ghostDis)

    if sum(foodDis):
        reverseFood = 1/sum(foodDis)

    score = currentGameState.getScore() + notEaten + reverseFood
    if sum(scaredTimes):
        score += scaredSum + (-5) * capsules + (-1) * minGhostDis 
        #  + (-5) * minFoodDis
    else:
        score += ghostDisSum + capsules 

    return score
    # util.raiseNotDefined()



# Abbreviation
better = betterEvaluationFunction
