import sys
import time
from IN104_simulateur.gameState import *
from IN104_simulateur.move import *
from random import randint
from IN104_simulateur.cell import Cell
from IN104_simulateur.game import *

INFINI = 10000000
MAX_STATE = 0
MIN_STATE = 1

class TimerAlphaBetaBrain:
    def __init__(self):
        self.name = "timeAlphaBetaBrain"
        self.computingTimes = []
        self.maxTime = 200
        config = {  'nRows': 8, # size of the board
                    'nPieces': 12, # number of pieces at the beginning of the game (should be a multiple of nRows < nRows**2/2)
                    'whiteStarts': True,
                 }
        game = Game(self, 10000, self, 10000, {'nRows':8, 'nPieces': 12}, 1000)

        self.timeMovement = self.timer(game.gameState)


    def timer(self,gameState):
        T=0
        for i in range(15):
            start = time.time()
            stateDict = gameState.findNextStates()
            end = time.time()
            if T < end - start:
                T = end - start
            gameState = list(stateDict.values())[0]

        start = time.time()
        self.eval(gameState)
        end = time.time()
        T = T + end - start
        print(T)
        return T

    def play(self, gameState, timeLimit):
        possibleMoves = gameState.getStateMoveDict()
        print("Authorized moves : ")
        for m in possibleMoves.values(): print(m.toPDN())
        string = ""
        try:
            weight, move = weight, move = self.alphaBeta(gameState, self.maxTime, MAX_STATE, -INFINI, INFINI, 0)
            print(self.name + " plays move : " + move.toPDN())

            choice = gameState.doMove(move, inplace = False)
            if str(choice) not in possibleMoves.keys(): raise Exception
        except Exception:
            print(string+' is an invalid move !')
            raise

        return choice

    def alphaBeta(self, gameState, timeLeft, state, alpha, beta, depth):
        """ alpha : best maximum on the path
            beta : best minimum on the path """

        moveDict = gameState.getStateMoveDict()
        if (timeLeft < self.timeMovement or not moveDict):
            #print("depth : ", depth)
            return self.eval(gameState),None
        possibleMoves = list(moveDict.values())

        maxWeight = -INFINI
        minWeight = INFINI
        for move in possibleMoves:
            nbrMoves = len(possibleMoves)
            newTime = (timeLeft-self.timeMovement)/nbrMoves
            if state == MAX_STATE:
                if alpha < beta:
                    newGameState = gameState.doMove(move)
                    curWeight, curMove = self.alphaBeta(newGameState, newTime, not state, alpha, beta, depth+1)
                    if curWeight > maxWeight:
                        maxWeight = curWeight
                        bestMove = move
                        alpha = max(alpha, maxWeight)
            elif state == MIN_STATE:
                if beta > alpha:
                    newGameState = gameState.doMove(move)
                    curWeight,curMove = self.alphaBeta(newGameState, newTime, not state, alpha, beta, depth+1)
                    if curWeight < minWeight:
                        minWeight = curWeight
                        bestMove = move
                        beta = min(beta, minWeight)

        if state == MAX_STATE:
            return maxWeight, bestMove
        elif state == MIN_STATE:
            return minWeight, bestMove

    def eval(self, gameState):
        bigWeight = 1000
        if not gameState.getStateMoveDict():
            hasWon = not gameState.isWhiteTurn
            if hasWon:
                return bigWeight
            else:
                return -bigWeight
        else:
            nbrWhites = 0
            nbrBlacks = 0
            for cell in gameState.boardState.cells:
                if cell.isBlack():
                     nbrBlacks += 1
                elif cell.isWhite():
                     nbrWhites += 1
            return nbrWhites - nbrBlacks



    def __str__(self):
        return self.name
