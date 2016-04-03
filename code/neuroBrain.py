import sys
import time
import random

from IN104_simulateur.gameState import *
from IN104_simulateur.move import *
from random import randint
from IN104_simulateur.cell import Cell

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

class NeuroBrain:
    def __init__(self, name):
        self.name = name
        self.computingTimes = []
        self.U = np.array([0]).reshape(1,1)

        # Neural Net constants
        self.model = None
        self.createNeuralNetwork()
        self.gamma = 0.9 # since it may take several moves to goal, making gamma high
        self.epsilon = 0.3 # epsilon-greedy algorithm
        self.step = "test"

    def createNeuralNetwork(self):
        self.model = Sequential()
        self.model.add(Dense(164, init='lecun_uniform', input_shape=(164,)))
        self.model.add(Activation('relu'))
        #self.model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        self.model.add(Dense(150, init='lecun_uniform'))
        self.model.add(Activation('relu'))
        #self.model.add(Dropout(0.2))

        self.model.add(Dense(1, init='lecun_uniform'))
        self.model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)

    def play(self, gameState, timeLimit):
        possibleMoves = gameState.getStateMoveDict()
        print("Authorized moves : ")
        for m in possibleMoves.values(): print(m.toPDN())
        string = ""
        while True:
            try:
                if self.step == "train":
                    string = self.nextMoveTrain(gameState)
                else:
                    string = self.nextMoveTest(gameState)
                print(self.name + " plays move : " + string)

                move = Move.fromPDN(string)
                choice = gameState.doMove(move, inplace = False)
                if str(choice) not in possibleMoves.keys(): raise Exception
                break
            except Exception:
                print(string+' is an invalid move !')
                raise

        return choice

    def nextMoveTrain(self, gameState):
        possibleMoves = list(gameState.getStateMoveDict().values())
        U = self.predict(gameState)
        print ("U : " + str(U))

        if (random.random() < self.epsilon): #choose random action
            action = possibleMoves[np.random.randint(0,len(possibleMoves))]
        else:
            newU = []
            for action in possibleMoves:
                newGameState = gameState.doMove(action)
                U_a = self.predict(newGameState)
                newU.append(self.getReward(gameState.doMove(action)) + self.gamma * U_a)
            print("New U : ", end="")
            print(newU)
            action = possibleMoves[np.argmax(newU)]

        print("Action selected : " + str(action.toPDN()))
        newGameState = gameState.doMove(action)
        reward = self.getReward(newGameState)
        newU = self.predict(newGameState)
        if reward == -1:
            update = reward + self.gamma * newU
        else:
            update = reward
        y = np.array([update]).reshape(1,1)

        print("Fitting...")
        self.fit(gameState, y)
        #time.sleep(0.04)
        print ("epsilon : " + str(self.epsilon))

        return action.toPDN()

    def nextMoveTest(self, gameState):
        possibleMoves = list(gameState.getStateMoveDict().values())
        U = self.predict(gameState)
        print ("U : " + str(U))

        newU = []
        for action in possibleMoves:
            newGameState = gameState.doMove(action)
            U_a = self.predict(newGameState)
            newU.append(self.getReward(gameState.doMove(action)) + self.gamma * U_a)
        print("New U : ", end="")
        print(newU)
        action = possibleMoves[np.argmax(newU)]

        print("Action selected : " + str(action.toPDN()))
        newGameState = gameState.doMove(action)
        reward = self.getReward(newGameState)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))

        return action.toPDN()

    def getReward(self, gameState):
        winningReward = 100
        if not gameState.getStateMoveDict():
            hasWon = not gameState.isWhiteTurn
            if hasWon:
                return winningReward
            else:
                return -winningReward
        else:
            return -1

    def getInput(self,gameState):
        listCells = gameState.boardState.cells
        tInput = []
        nbrWhites, nbrBlacks, nbrKingsBlack, nbrKingsWhite = 0,0,0,0
        for cell in listCells:
            if cell == Cell.empty:
                tInput.append([1, 0, 0, 0, 0])
            if cell == Cell.b:
                nbrBlacks += 1
                tInput.append([0, 1, 0, 0, 0])
            if cell == Cell.B:
                nbrBlacks += 1
                nbrKingsBlack += 1
                tInput.append([0, 0, 1, 0, 0])
            if cell == Cell.w:
                nbrWhites += 1
                tInput.append([0, 0, 0, 1, 0])
            if cell == Cell.W:
                nbrKingsWhite += 1
                nbrWhites += 1
                tInput.append([0, 0, 0, 0, 1])
        tInput = np.array(tInput).reshape(160)
        tInput = np.concatenate([tInput, [nbrWhites], [nbrBlacks], [nbrKingsWhite], [nbrKingsBlack]])
        return tInput.reshape(1,164)

    def predict(self, gameState):
        return self.model.predict(self.getInput(gameState), batch_size=1)

    def fit(self, gameState, y):
        return self.model.fit(self.getInput(gameState), y, batch_size=1, nb_epoch=1, verbose=1)

    def saveWeights(self, filename='weights.h5'):
        self.model.save_weights(filename)

    def loadWeights(self, filename='weights.h5'):
        self.model.load_weights(filename)

    def __str__(self):
        return self.name
