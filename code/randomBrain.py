import sys
import time
from IN104_simulateur.gameState import *
from IN104_simulateur.move import *
from random import randint

class RandomBrain:
    def __init__(self):
        self.name = "RandomBrain"
        self.computingTimes = []

    def play(self, gameState, timeLimit):
        possibleMoves = gameState.getStateMoveDict()
        print("Authorized moves : ")
        for m in possibleMoves.values(): print(m.toPDN())
        string = ""
        while True:
            try:
                print(self.name + " plays move : ", end=" ")
                #time.sleep(1)
                string = self.randomString(list(possibleMoves.values()))
                print(string)

                move = Move.fromPDN(string)
                choice = gameState.doMove(move, inplace = False)
                if str(choice) not in possibleMoves.keys(): raise Exception
                break
            except Exception as e:
                print(string+' is an invalid move !')
                print (e)

        return choice

    def randomString(self, possibleMovesList):
        coup = randint(0,len(possibleMovesList)-1)
        return possibleMovesList[coup].toPDN()

    def __str__(self):
        return self.name
