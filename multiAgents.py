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


from util import manhattanDistance
from game import Directions
import random, util

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

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newGhostPos = successorGameState.getGhostPositions()

        newx = newPos[0]
        newy = newPos[1]
        distFoods = [(abs(x-newx) + abs(y-newy)) for x,y in newFood]
        distGhosts = [(abs(int(x)-newx) + abs(int(y)-newy)) for x,y in newGhostPos]

        #New score
        newscore = 0
        newscore = successorGameState.getScore() - currentGameState.getScore()
        newscore = newscore*2

        #print('ScoreFactor:')
        #print(newscore)

        #Meal Magic
        foodFactor = 0
        for foodDist in distFoods:
            foodFactor += 10/foodDist

        #print('FoodFactor:')
        #print(foodFactor)
        
        #Ghost Factor
        distClosestGhost = 0

        for distGhost in distGhosts:
            if (distGhost < 5):
                distClosestGhost += (5 - distGhost)

        ghostFear = distClosestGhost*1.5
        
        i = 1
        for times in newScaredTimes:
            i = i * times
        if i > 0:
            ghostFear = 0

        #print('ghostFear:')
        #print(ghostFear)
        
        return newscore + foodFactor - ghostFear

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        numAgents = gameState.getNumAgents()
        #print("Num Agents:")
        #print(numAgents)

        #print(gameState.getLegalActions(0))
        #print(gameState.getLegalActions(1))

        res = value(gameState, 0, self.depth, numAgents, self)

        act = res[1]

        return act
    
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        res = prune_value(gameState, 0, self.depth, numAgents, self, -100000, 100000)

        act = res[1]

        return act

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
        numAgents = gameState.getNumAgents()

        res = exp_value(gameState, 0, self.depth, numAgents, self)

        act = res[1]

        return act

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I used my code from question 1 which estimates the state's value based on a combination of features.
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    newGhostPos = currentGameState.getGhostPositions()

    newx = newPos[0]
    newy = newPos[1]
    distFoods = [(abs(x-newx) + abs(y-newy)) for x,y in newFood]
    distGhosts = [(abs(int(x)-newx) + abs(int(y)-newy)) for x,y in newGhostPos]

    #New score
    newscore = 0
    newscore = currentGameState.getScore() 
    newscore = newscore*2.5

    #print('ScoreFactor:')
    #print(newscore)

    #Meal Magic
    foodFactor = 0
    for foodDist in distFoods:
        foodFactor += 13/foodDist

    #print('FoodFactor:')
    #print(foodFactor)
        
    #Ghost Factor
    distClosestGhost = 0

    for distGhost in distGhosts:
        if (distGhost < 7):
            distClosestGhost += (5 - distGhost)

    ghostFear = distClosestGhost*1.3
        
    i = 1
    for times in newScaredTimes:
        i = i * times
    if i > 0:
        ghostFear = 0

    #print('ghostFear:')
    #print(ghostFear)
        
    return newscore + foodFactor - ghostFear
    

#Minmax
def value(gameState, curAgent, depth, numAgent, agent):
    if gameState.isWin():
        return (agent.evaluationFunction(gameState),'')
    if gameState.isLose():
        return (agent.evaluationFunction(gameState),'')
    if depth == 0:
        return (agent.evaluationFunction(gameState),'')
    else:
        if curAgent == 0:
            return max_value(gameState, curAgent, depth, numAgent, agent)
        else:
            return min_value(gameState, curAgent, depth, numAgent, agent)

def max_value(gameState, curAgent, depth, numAgent, agent):
    nextAgent = (curAgent + 1) % numAgent
    #print ("Cur agent max:")
    #print(curAgent)
    #print(gameState.getLegalActions(curAgent))
        
    if (nextAgent < curAgent):
        nextdepth = depth - 1
    else:
        nextdepth = depth
            
    i = 0
        
    for action in gameState.getLegalActions(curAgent):
        res = value(gameState.generateSuccessor(curAgent, action), nextAgent, nextdepth, numAgent, agent)
        val = res[0]
        if i == 0:
            maxval = val
            maxact = action

        if val > maxval:
            maxval = val
            maxact = action
                
        i = i + 1
            
    return (maxval, maxact)

def min_value(gameState, curAgent, depth, numAgent, agent):
    nextAgent = (curAgent + 1) % numAgent
    #print ("Cur agent min:")
    #print(curAgent)
    #print(gameState.getLegalActions(curAgent-1))
    #print(gameState.getLegalActions(curAgent))
    #print(gameState.isLose())
        
    if (nextAgent < curAgent):
        nextdepth = depth - 1
    else:
        nextdepth = depth
            
    i = 0
        
    for action in gameState.getLegalActions(curAgent):
        res = value(gameState.generateSuccessor(curAgent, action), nextAgent, nextdepth, numAgent, agent)
        #print("Res:")
        #print(res)
        val = res[0]
        if i == 0:
            minval = val
            minact = action

        if val < minval:
            minval = val
            minact = action
                
        i = i + 1

    return (minval, minact)

#Minmax with pruning
def prune_value(gameState, curAgent, depth, numAgent, agent, alpha, beta):
    if gameState.isWin():
        return (agent.evaluationFunction(gameState),'')
    if gameState.isLose():
        return (agent.evaluationFunction(gameState),'')
    if depth == 0:
        return (agent.evaluationFunction(gameState),'')
    else:
        if curAgent == 0:
            return prune_max_value(gameState, curAgent, depth, numAgent, agent, alpha, beta)
        else:
            return prune_min_value(gameState, curAgent, depth, numAgent, agent, alpha, beta)

def prune_max_value(gameState, curAgent, depth, numAgent, agent, alpha, beta):
    nextAgent = (curAgent + 1) % numAgent
    #print ("Cur agent max:")
    #print(curAgent)
    #print(gameState.getLegalActions(curAgent))
        
    if (nextAgent < curAgent):
        nextdepth = depth - 1
    else:
        nextdepth = depth
            
    i = 0
        
    for action in gameState.getLegalActions(curAgent):
        res = prune_value(gameState.generateSuccessor(curAgent, action), nextAgent, nextdepth, numAgent, agent, alpha, beta)
        val = res[0]
        if i == 0:
            maxval = val
            maxact = action

        if val > beta:
            return (val, action)

        if val > maxval:
            maxval = val
            maxact = action

        if val > alpha:
            alpha = val
                
        i = i + 1
            
    return (maxval, maxact)

def prune_min_value(gameState, curAgent, depth, numAgent, agent, alpha, beta):
    nextAgent = (curAgent + 1) % numAgent
    #print ("Cur agent min:")
    #print(curAgent)
    #print(gameState.getLegalActions(curAgent-1))
    #print(gameState.getLegalActions(curAgent))
    #print(gameState.isLose())
        
    if (nextAgent < curAgent):
        nextdepth = depth - 1
    else:
        nextdepth = depth
            
    i = 0
        
    for action in gameState.getLegalActions(curAgent):
        res = prune_value(gameState.generateSuccessor(curAgent, action), nextAgent, nextdepth, numAgent, agent, alpha, beta)
        #print("Res:")
        #print(res)
        val = res[0]
        if i == 0:
            minval = val
            minact = action

        if val < alpha:
            return (val, action)

        if val < minval:
            minval = val
            minact = action

        if val < beta:
            beta = val
                
        i = i + 1

    return (minval, minact)

#Expectimax
#Minmax
def exp_value(gameState, curAgent, depth, numAgent, agent):
    if gameState.isWin():
        return (agent.evaluationFunction(gameState),'')
    if gameState.isLose():
        return (agent.evaluationFunction(gameState),'')
    if depth == 0:
        return (agent.evaluationFunction(gameState),'')
    else:
        if curAgent == 0:
            return exp_max_value(gameState, curAgent, depth, numAgent, agent)
        else:
            return exp_min_value(gameState, curAgent, depth, numAgent, agent)

def exp_max_value(gameState, curAgent, depth, numAgent, agent):
    nextAgent = (curAgent + 1) % numAgent
    #print ("Cur agent max:")
    #print(curAgent)
    #print(gameState.getLegalActions(curAgent))
        
    if (nextAgent < curAgent):
        nextdepth = depth - 1
    else:
        nextdepth = depth
            
    i = 0
        
    for action in gameState.getLegalActions(curAgent):
        res = exp_value(gameState.generateSuccessor(curAgent, action), nextAgent, nextdepth, numAgent, agent)
        val = res[0]
        if i == 0:
            maxval = val
            maxact = action

        if val > maxval:
            maxval = val
            maxact = action
                
        i = i + 1
            
    return (maxval, maxact)

def exp_min_value(gameState, curAgent, depth, numAgent, agent):
    nextAgent = (curAgent + 1) % numAgent
    #print ("Cur agent min:")
    #print(curAgent)
    #print(gameState.getLegalActions(curAgent-1))
    #print(gameState.getLegalActions(curAgent))
    #print(gameState.isLose())
        
    if (nextAgent < curAgent):
        nextdepth = depth - 1
    else:
        nextdepth = depth
            
    i = 0

    actionvalue = 0
        
    for action in gameState.getLegalActions(curAgent):
        res = exp_value(gameState.generateSuccessor(curAgent, action), nextAgent, nextdepth, numAgent, agent)
        #print("Res:")
        #print(res)
        val = res[0]

        actionvalue = actionvalue + val

        
        if i == 0:
            minval = val
            minact = action

        if val < minval:
            minval = val
            minact = action
                
        i = i + 1

    return (actionvalue/i, '')

# Abbreviation
better = betterEvaluationFunction
