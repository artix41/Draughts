import sys
import random
import time

from IN104_simulateur.game import *
from IN104_simulateur.cell import Cell
from manualBrain import *
from randomBrain import *
from minMaxBrain import *
from neuroBrain import *

def test():
    config = {  'nRows': 8, # size of the board
                'nPieces': 12, # number of pieces at the beginning of the game (should be a multiple of nRows < nRows²/2)
                'whiteStarts': True,
             }

    ia1 = NeuroBrain("NeuroBrain")
    ia2 = RandomBrain("Random Guesser")

    ia1.loadWeights()
    start = time.time()
    nbrWin = 0
    for i in range(1):
        print ("=================== Game " + str(i) + " ================")

        game = Game(ia1, 10000, ia2, 10000, config, 1000) # syntax : Game(ia1, ia2, config [, Nlimit = 150])
        #game.player1.showTime = True # show the time spent by an IA
        #game.player1.timeLimit = 1 # you can change timeLimit per player (unfair game)

        # displayLevel of the game :
        # 0/ does not display anything
        # 1/ displays the board state evolution plus error messages
        # 2/ also displays the list of possible moves
        # 3/ displays everything that is put into the logs
        game.displayLevel = 1
        game.pause = 0
        game.gameState.boardState.debug = False

        pdn = game.runGame()

    ia1.saveWeights()

    # Save logs and pdn in text files
    print ("======================= Time ========================")
    print (time.time() - start)

    return

    import datetime as dt
    s = str(dt.datetime.today())
    fileName = str(ia1)+"_vs_"+str(ia2)+"_"+s[s.find(' ')+1:s.find('.')]
    logFile = 'logs/'+fileName+'.log'
    pdnFile = 'pdns/'+fileName+'.pdn'
    with open(logFile, "w") as f:
        f.write(game.log)
    if pdn:
        with open(pdnFile, "w") as f:
            f.write(pdn)
    #'''

    # plot the computation times
    import matplotlib.pylab as plt
    plt.plot(game.player1.computingTimes,'blue')
    plt.plot(game.player2.computingTimes,'red')
    plt.show()


if __name__ == '__main__':
    test()
