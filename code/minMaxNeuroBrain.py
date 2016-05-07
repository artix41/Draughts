import sys
import time
from IN104_simulateur.gameState import *
from IN104_simulateur.move import *
from random import randint, random
from IN104_simulateur.cell import Cell

import numpy as np
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

INFINI = 10000000
MAX_STATE = 0
MIN_STATE = 1

class MinMaxNeuroBrain:
    def __init__(self):
        self.name = "minMaxNeuroBrain"
        self.computingTimes = []
        self.model = self.loadNN("model.yaml", "weights.h5")


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

    def loadNN(self, model_file, weights_file):
        with open(model_file, "r") as f:
            yaml_string = f.read()
            model = model_from_yaml(yaml_string)
        model.load_weights(weights_file)
        return model

    def eval(self, gameState):
        return self.model.predict(gameState)


    def __str__(self):
        return self.name
