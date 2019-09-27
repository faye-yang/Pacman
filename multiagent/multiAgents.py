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


import random
import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

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

        # find the cloest ghost
        close_ghost_dis = 0
        for each_ghost in newGhostStates:
            ghost_distance = manhattanDistance(newPos, each_ghost.getPosition())
            if close_ghost_dis == 0:
                close_ghost_dis = ghost_distance
            if ghost_distance < close_ghost_dis:
                close_ghost_dis = ghost_distance
        if close_ghost_dis < 2:
            return float("-inf")
        #find the closet food
        closet_food = 0
        for food in newFood.asList():
            food_distance = manhattanDistance(newPos, food)
            if closet_food == 0:
                closet_food = food_distance
            if food_distance < closet_food:
                closet_food = food_distance


        return successorGameState.getScore() - 1/close_ghost_dis +10/(closet_food+1)


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
      player1 stand for pacman
      player0 stand for ghost
    """


    def MinMax(self,gameState, depth,ghostNum, player= 1):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [None, self.evaluationFunction(gameState)]
            # current depth

        new_state = []
        if player == 1:
            for action in gameState.getLegalActions(0):
                new_state.append([action, (self.MinMax(gameState.generateSuccessor(0, action), depth,1, 0))[1]])
           #find the max result
            max_result = []
            for state in new_state:
                if max_result == []:
                    max_result = state
                if state[1] > max_result[1]:
                    max_result = state
            return max_result

        else:
            for action in gameState.getLegalActions(ghostNum):
               #if there is ghost
                if ghostNum != (gameState.getNumAgents() - 1):
                    new_state.append([action,
                    self.MinMax(gameState.generateSuccessor(ghostNum, action), depth,ghostNum + 1,0)[1]])
               #go through all the ghost
                if ghostNum == (gameState.getNumAgents() - 1):
                    new_state.append(
                        [action, (self.MinMax(gameState.generateSuccessor(ghostNum, action), depth - 1,ghostNum,1))[1]])

            #find the min result
            min_result = []
            for state in new_state:
                if min_result == []:
                    min_result = state
                if state[1] < min_result[1]:
                    min_result = state

            return min_result


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
        """
        "*** YOUR CODE HERE ***"

        depth = self.depth
        move=self.MinMax(gameState, depth,1,1)
        return move[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def AlphaBeta(self,gameState, depth,ghostNum,alpha=-float("inf"), beta=float("inf"),  player=1):

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return [None, self.evaluationFunction(gameState)]
        new_state = []
        if player == 1:
            max_result = []
            for action in gameState.getLegalActions(0):
                new_state.append([action, (self.AlphaBeta(gameState.generateSuccessor(0, action), depth,1,alpha,beta, 0))[1]])
               #find the max result
                for state in new_state:
                    if max_result == []:
                        max_result = state
                    if state[1] > max_result[1]:
                        max_result = state
                alpha = max(alpha, max_result[1])
                if beta <= alpha:
                    break
            return max_result

        else:
            min_result = []
            for action in gameState.getLegalActions(ghostNum):

                if ghostNum != (gameState.getNumAgents() - 1):
                    new_state.append([action,
                    self.AlphaBeta(gameState.generateSuccessor(ghostNum, action), depth,ghostNum + 1,alpha,beta,0)[1]])
                #go through all the ghost
                if ghostNum == (gameState.getNumAgents() - 1):
                    new_state.append(
                        [action, (self.AlphaBeta(gameState.generateSuccessor(ghostNum, action), depth - 1,ghostNum,alpha,beta,1))[1]])

                for state in new_state:
                    if min_result == []:
                        min_result = state
                    if state[1] < min_result[1]:
                        min_result = state
                beta = min(beta, min_result[1])
                if beta <= alpha:
                    break

            return min_result

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth=self.depth
        result=self.AlphaBeta(gameState, depth, 1)
        return result[0]


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
        depth = self.depth
        result = self.max(gameState, depth)
        return result[0]

    def max(self, gameState, depth):

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return [None, self.evaluationFunction(gameState)]
        new_state = []
       #find all the value
        for action in gameState.getLegalActions(0):
            new_state.append([action, (self.chance(gameState.generateSuccessor(0, action), depth, 1))])

        #pick the highest value
        max_result = []
        for state in new_state:
            if max_result == []:
                max_result = state
            if state[1] > max_result[1]:
                max_result = state

        return max_result

    def chance(self, gameState, depth, ghostNum):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        new_state = []
        for action in gameState.getLegalActions(ghostNum):
           #still is ghost recrusive
            if ghostNum != (gameState.getNumAgents() - 1):
                new_state.append((self.chance(gameState.generateSuccessor(ghostNum, action), depth, ghostNum + 1)))
            #no ghost anymore (gameState.getNumAgents()-1=# of ghost)
            if ghostNum == (gameState.getNumAgents() - 1):
                new_state.append((self.max(gameState.generateSuccessor(ghostNum, action), depth - 1))[1])


        # find the chance by assuming it is uniform distribution
        sum = 0
        for state in new_state:
            sum += state

        return float(sum)/float(len(new_state))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      1. pacman go to the closest food
      2. do not meet any ghost. to avoid ending/lossing game easily. return - infinity if the pacman across any ghost.
      3. Pacman  eat capsule and pacman should find the capsule
      4. Pacman eat good ghosts,give a 80 / distance from Pacman to the good ghosts matters

    """
    position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    curghost = currentGameState.getGhostStates()
    capsule_list = currentGameState.getCapsules()
    score = currentGameState.getScore()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curghost]

    #find the distance to the cloest food
    closet_food = 0
    for each_food in foods.asList():
        food_distance = manhattanDistance(position, each_food)
        if closet_food == 0:
            closet_food = food_distance
        if food_distance < closet_food:
            closet_food = food_distance

    #find the closet capsule
    closet_capsule = 0
    for each_capsule in capsule_list:
        capsule_distance = manhattanDistance(position, each_capsule)
        if capsule_distance == 0:
            closet_capsule = capsule_distance
        if capsule_distance < closet_capsule:
            closet_capsule = capsule_distance

    #find the cloest ghost
    close_ghost_dis = 0
    for each_ghost in curghost:
        ghost_distance = manhattanDistance(position, each_ghost.getPosition())
        if close_ghost_dis == 0:
            close_ghost_dis = ghost_distance
        if ghost_distance < close_ghost_dis:
            close_ghost_dis = ghost_distance

    #avoid ghost
    if close_ghost_dis < 2:
        return float("-inf")


    # Pacman eat good ghosts, giving a bonus
    numGoodGhost = 0
    for timer in range(len(curScaredTimes)):
        dist = manhattanDistance(position, curghost[timer].getPosition())
        if curScaredTimes[timer] != 0 and dist > 0:
            numGoodGhost += 80.0 / dist
    return score+10/(closet_food+1)+10/(closet_capsule+1) - 1/close_ghost_dis+numGoodGhost

# Abbreviation
better = betterEvaluationFunction
