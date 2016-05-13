import sys
import time
import random

from IN104_simulateur.gameState import *
from IN104_simulateur.move import *
from random import randint
from IN104_simulateur.cell import Cell

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop

INFINI = 100000000

class DeepNeuroBrain:
    def __init__(self, step="test"):
        """ Input : step : 'train' or 'test', depending if you want to fit the
        neural network or not at each game."""

        self.name = "neuroBrain"
        self.computingTimes = []
        self.U = np.array([0]).reshape(1,1) # evaluation function, output of the NN

        # Neural Net constants
        self.model = None
        self.createNeuralNetwork()
        self.gamma = 0.9 # discount factor
        self.epsilon = 0.1 # epsilon-greedy algorithm
        self.normalReward = -1 # reward in normal games (not winning or losing)

        self.winningReward = 100
        self.losingReward = -100
        self.step = step
        self.verbose = 0 # if 0, only print the value of U at each game

    def createNeuralNetwork(self):
        """ Create and compile a convolutional neural network with Keras """

        print("Create the neural network...")
        self.model = Sequential()
        self.model.add(Convolution2D(32, 4, 4, border_mode='same', input_shape=(1,8, 8)))
        self.model.add(Convolution2D(16, 4, 4, border_mode='same', input_shape=(1,8, 8)))

        self.model.add(Flatten())

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
        """ Given an ennemi game state, compute the best move for him,
        ie with the worst U, and return the couple (U,reward) corresponding """

        if not gameState.getStateMoveDict():
            reward = self.getReward(gameState) # either win or draw, because the ennemi (gameState) has no possible move
            return (reward,reward)
        possibleMoves = list(gameState.getStateMoveDict().values())
        minU = INFINI
        for action in possibleMoves:
            newGameState = gameState.doMove(action)
            reward = self.getReward(newGameState)
            if not newGameState.getStateMoveDict():
                return (reward,reward) # either lose or draw
            if reward + self.gamma * self.predict(newGameState) < minU:
                minU = reward + self.gamma * self.predict(newGameState)
                minReward = reward
        return (minU,self.normalReward)

    def getListNextUR(self, gameState, possibleMoves):
        """ Given our gameState and a list of possibleMoves, return a list of
        the U functions and rewards corresponding to all our moves. To compute
        the U function after a move, as the new board is ennemi, we consider the
        U of his best move (by calling the function getMinUR). It's a sort of
        deep-2 minMax. """

        listU = []
        listR = []
        for action in possibleMoves:
            newGameState = gameState.doMove(action)
            newU, newR = self.getMinUR(newGameState)
            listU.append(newU)
            listR.append(newR)
        return (listU, listR)

    def nextMoveTrain(self, gameState):
        """ Reinforcement learning algorithm (TD(0)) with epsilon-greedy to chose the action
        Perform a sort of min-max with deep 2 to determine the best action
        (for each action, the eval function (called U) of the new game state is the eval function
        of the game state after the ennemi took his best move)
        The U function is the result of a neural network, and is updated after each move """

        possibleMoves = list(gameState.getStateMoveDict().values())
        U = self.predict(gameState)
        print ("U : " + str(U), end="")

        newU = []
        if (random.random() < self.epsilon): #choose random action (epsilon-greedy part)
            print(" : random")
            action = possibleMoves[np.random.randint(0,len(possibleMoves))]
            newGameState = gameState.doMove(action)
            newU, reward = self.getMinUR(newGameState) # newU corresponds to the best move of the ennemi after we took the random action
        else:
            print("")
            newUR = self.getListNextUR(gameState, possibleMoves) # newUR = (listOfU, listOfReward)
            iBestMove = np.argmax(newUR[0]) # We take the best action (with the best U)
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
            update = reward + self.gamma * newU # updated U, according to TD(0) algorithm
        y = np.array([update]).reshape(1,1)

        if self.verbose:
            print("Update : " + str(update))
            print("Fitting...")

        self.fit(gameState, y)
        #time.sleep(0.04)

        return action.toPDN()

    def nextMoveTest(self, gameState):
        """ Same than nextMoveTrain, but without fitting with an update """
        possibleMoves = list(gameState.getStateMoveDict().values())
        U = self.predict(gameState)
        print ("U : " + str(U))

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
        """ Turn the gameState into the format given to the input of the NN """
        listCells = gameState.boardState.cells
        tInput = np.zeros((8,8))
        nbrWhites, nbrBlacks, nbrKingsBlack, nbrKingsWhite = 0,0,0,0
        iterCell = listCells.__iter__()
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    cell = iterCell.__next__()
                    if cell == Cell.empty:
                        tInput[row,col] = 0
                    if cell == Cell.b:
                        tInput[row,col] = -1
                    if cell == Cell.B:
                        tInput[row,col] = -3
                    if cell == Cell.w:
                        tInput[row,col] = 1
                    if cell == Cell.W:
                        tInput[row,col] = 3
        return tInput.reshape(1,1,8,8)

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
