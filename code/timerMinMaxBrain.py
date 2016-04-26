import sys
import time
from IN104_simulateur.gameState import *
from IN104_simulateur.move import *
from random import randint
from IN104_simulateur.cell import Cell

INFINI = 10000000
MAX_STATE = 0
MIN_STATE = 1

class MinMaxBrain:
    def __init__(self, name):
        self.name = name
        self.computingTimes = []
        self.maxDeep = 2
        self.timeson = self.timer(gameState)
        
        
    def timer(self,gameState):
        T=0
        for i in range(10):
            start = time.time()
            stateDict = gameState.findNextStates
            end = time.end()
            if T < start-end : T = start-end
            gameState = gameState.findNextStates.values()[0]
            
        start = time.time()
        self.eval(gameState)
        end = time.end()
        T = T + start-end
        print(T)
        return T
        
    def play(self, gameState, timeLimit):
        possibleMoves = gameState.getStateMoveDict()
        print("Authorized moves : ")
        for m in possibleMoves.values(): print(m.toPDN())
        string = ""
        while True:
            try:
                weight, string = self.minMax(gameState, INFINI, 1, MAX_STATE)
                print(self.name + " plays move : " + string)

                move = Move.fromPDN(string)
                choice = gameState.doMove(move, inplace = False)
                if str(choice) not in possibleMoves.keys(): raise Exception
                break
            except Exception:
                print(string+' is an invalid move !')
                raise

        return choice

    def minMax(self, gameState, previousWeight, deep, state):
        """ Input : state : MAX_STATE or MIN_STATE """
        moveDict = gameState.getStateMoveDict()
        if (deep > self.maxDeep or not moveDict):
            return self.eval(gameState),None
        possibleMoves = list(moveDict.values())

        maxWeight = -INFINI
        minWeight = INFINI
        for move in possibleMoves:
            if state == MAX_STATE:
                if maxWeight <= previousWeight: # alpha-beta : we check if it's not useless to explore the branch
                    newGameState = gameState.doMove(move)
                    curWeight, curMove = self.minMax(newGameState, maxWeight, deep+1, not state)
                    if curWeight > maxWeight:
                        maxWeight = curWeight
                        bestMove = move
            elif state == MIN_STATE:
                if minWeight >= previousWeight:
                    newGameState = gameState.doMove(move)
                    curWeight,curMove = self.minMax(newGameState, maxWeight, deep+1, not state)
                    if curWeight < minWeight:
                        minWeight = curWeight
                        bestMove = move

        if state == MAX_STATE:
            return maxWeight, bestMove.toPDN()
        elif state == MIN_STATE:
            return minWeight, bestMove.toPDN()

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
