import sys
import random
import time
import matplotlib.pyplot as plt

from IN104_simulateur.game import *
from IN104_simulateur.cell import Cell
from manualBrain import *
from randomBrain import *
from minMaxBrain import *
from alphaBetaBrain import *
from timerAlphaBetaBrain import *
from minMaxNeuroBrain import *
from neuroBrain import *
from deepNeuroBrain import *

def test():
    config = {  'nRows': 8, # size of the board
                'nPieces': 12, # number of pieces at the beginning of the game (should be a multiple of nRows < nRows**2/2)
                'whiteStarts': True,
             }

    ia1 = NeuroBrain()
    ia2 = AlphaBetaBrain()

    ia1.loadWeights()
    start = time.time()
    nbrWin, nbrDraws, nbrLose = 0, 0, 0
    percentWin = []
    percentLose = []
    for i in range(20000):
        try:
            print ("=================== Game " + str(i) + " ================")

            game = Game(ia1, 10000, ia2, 10000, config, 1000) # syntax : Game(ia1, ia2, config [, Nlimit = 150])
            #game.player1.showTime = True # show the time spent by an IA
            #game.player1.timeLimit = 1 # you can change timeLimit per player (unfair game)

            # displayLevel of the game :
            # 0/ does not display anything
            # 1/ displays the board state evolution plus error messages
            # 2/ also displays the list of possible moves
            # 3/ displays everything that is put into the logs
            game.displayLevel = 0
            game.pause = 0
            game.gameState.boardState.debug = False

            pdn = game.runGame()
            #time.sleep(2)
            if pdn[-6:-3] == "0-1":
                print ("White lose...")
                nbrLose += 1
            elif pdn[-6:-3] == "1-0":
                print ("White Win !")
                nbrWin += 1
            else:
                print("Draw")
                nbrDraws += 1
            print("Score : " + str(nbrWin) + "/" + str(i+1))
            print("Draws : " + str(nbrDraws))
            percentWin.append(float(nbrWin)/float(i+1))
            percentLose.append(float(nbrLose)/float(i+1))
            ia1.saveWeights()
        except KeyboardInterrupt:
            import matplotlib.pyplot as plt
            plt.plot(percentWin, color="red")
            plt.plot(percentLose, color="blue")
            plt.savefig("stats.png")
            plt.show()
            break

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
