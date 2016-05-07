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

INFINI = 100000000

class NeuroBrain:
    def __init__(self):
        self.name = "neuroBrain"
        self.computingTimes = []
        self.U = np.array([0]).reshape(1,1)

        # Neural Net constants
        self.model = None
        self.createNeuralNetwork()
        self.gamma = 0.9 # since it may take several moves to goal, making gamma high
        self.epsilon = 0.2 # epsilon-greedy algorithm
        self.normalReward = -1
        self.winningReward = 100
        self.step = "train"
        self.verbose = 0

    def createNeuralNetwork(self):
        print("Create the neural network...")
        self.model = Sequential()
        self.model.add(Dense(100, activation="relu", init='lecun_uniform', input_shape=(134,)))

        self.model.add(Dense(50, activation="relu", init='lecun_uniform'))

        self.model.add(Dense(25, activation="relu", init='lecun_uniform'))

        self.model.add(Dense(1, init='lecun_uniform'))

        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)
        yaml_string = self.model.to_yaml()
        with open("model.yaml", "w") as f:
            f.write(yaml_string)
        print("[+] Neural network created")

    def play(self, gameState, timeLimit):
        possibleMoves = gameState.getStateMoveDict()
        if self.verbose:
            print("Authorized moves : ")
            for m in possibleMoves.values(): print(m.toPDN())
        string = ""
        try:
            if self.step == "train":
                string = self.nextMoveTrain(gameState)
            else:
                string = self.nextMoveTest(gameState)

            move = Move.fromPDN(string)
            choice = gameState.doMove(move, inplace = False)
            if str(choice) not in possibleMoves.keys(): raise Exception
        except Exception:
            print(string+' is an invalid move !')
            raise

        return choice

    def getMinUR(self, gameState):
        if not gameState.getStateMoveDict():
            return (self.winningReward,self.winningReward)
        possibleMoves = list(gameState.getStateMoveDict().values())
        minU = INFINI
        for action in possibleMoves:
            newGameState = gameState.doMove(action)
            reward = self.getReward(newGameState)
            if reward == -self.winningReward:
                return (-self.winningReward,-self.winningReward)
            if reward + self.gamma * self.predict(newGameState) < minU:
                minU = reward + self.gamma * self.predict(newGameState)
                minReward = reward
        return (minU,self.normalReward)

    def getListNextUR(self, gameState, possibleMoves):
        listU = []
        listR = []
        for action in possibleMoves:
            newGameState = gameState.doMove(action)
            newU, newR = self.getMinUR(newGameState)
            listU.append(newU)
            listR.append(newR)
        return (listU, listR)

    def nextMoveTrain(self, gameState):
        possibleMoves = list(gameState.getStateMoveDict().values())
        U = self.predict(gameState)
        if self.verbose or True:
            print ("U : " + str(U))

        newU = []
        if (random.random() < self.epsilon): #choose random action
            action = possibleMoves[np.random.randint(0,len(possibleMoves))]
            newGameState = gameState.doMove(action)
            newU, reward = self.getMinUR(newGameState)
        else:
            newUR = self.getListNextUR(gameState, possibleMoves) # newUR = (listOfU, listOfReward)
            iBestMove = np.argmax(newUR[0])
            if self.verbose:
                print("New UR : ", newUR)
                print("iBestMove : ", iBestMove)
            reward = newUR[1][iBestMove]
            newU = newUR[0][iBestMove]
            action = possibleMoves[iBestMove]

        if self.verbose:
            print("Action selected : " + str(action.toPDN()))

        if reward != self.normalReward:
            update = reward
        else:
            update = reward + self.gamma * newU
        y = np.array([update]).reshape(1,1)

        if self.verbose:
            print("Update : " + str(update))
            print("Fitting...")

        self.fit(gameState, y)
        #time.sleep(0.04)

        return action.toPDN()

    def nextMoveTest(self, gameState):
        possibleMoves = list(gameState.getStateMoveDict().values())
        U = self.predict(gameState)
        print ("U : " + str(U))

        newUR = self.getListNextUR(gameState, possibleMoves) # newUR = (listOfU, listOfReward)
        print("New UR : ", end="")
        print(newUR)
        iBestMove = np.argmax(newUR[0])
        reward = newUR[1][iBestMove]
        newU = newUR[0][iBestMove]
        action = possibleMoves[iBestMove]
        print("Action selected : " + str(action.toPDN()))

        return action.toPDN()

    def getReward(self, gameState):
        if not gameState.getStateMoveDict():
            hasWon = not gameState.isWhiteTurn
            if hasWon:
                return self.winningReward
            else:
                return -self.winningReward
        else:
            return self.normalReward

    def getInput(self,gameState):
        listCells = gameState.boardState.cells
        tInput = []
        nbrWhites, nbrBlacks, nbrKingsBlack, nbrKingsWhite = 0,0,0,0
        for cell in listCells:
            if cell == Cell.empty:
                tInput.append([0, 0, 0, 0])
            if cell == Cell.b:
                nbrBlacks += 1
                tInput.append([1, 0, 0, 0])
            if cell == Cell.B:
                nbrBlacks += 1
                nbrKingsBlack += 1
                tInput.append([ 0, 1, 0, 0])
            if cell == Cell.w:
                nbrWhites += 1
                tInput.append([ 0, 0, 1, 0])
            if cell == Cell.W:
                nbrKingsWhite += 1
                nbrWhites += 1
                tInput.append([0, 0, 0, 1])
        tInput = np.array(tInput).reshape(128)
        tInput = np.concatenate([tInput, [nbrWhites], [nbrBlacks], [nbrKingsWhite], [nbrKingsBlack]])
        tInput = np.concatenate([tInput, [nbrWhites + nbrBlacks + nbrKingsWhite + nbrKingsBlack]])
        tInput = np.concatenate([tInput, [nbrWhites - nbrBlacks + 3*(nbrKingsWhite - nbrKingsBlack)]])
        return tInput.reshape(1,134)

    def predict(self, gameState):
        return self.model.predict(self.getInput(gameState), batch_size=1)

    def fit(self, gameState, y):
        return self.model.fit(self.getInput(gameState), y, batch_size=1, nb_epoch=1, verbose=self.verbose)

    def saveWeights(self, filename='weights.h5'):
        self.model.save_weights(filename, overwrite=True)

    def loadWeights(self, filename='weights.h5'):
        self.model.load_weights(filename)

    def __str__(self):
        return self.name
