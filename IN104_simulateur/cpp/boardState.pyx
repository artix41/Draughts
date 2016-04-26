# distutils: language = c++
# distutils: sources = IN104_simulateur/cpp/CBoardState.cpp IN104_simulateur/cpp/CCell.cpp IN104_simulateur/cpp/CMove.cpp
from cython.operator cimport dereference as deref, preincrement as inc

from .boardState cimport *

cdef class BoardState:
    '''
    This class is a Pyhton interface (front-end) for the C++ backend of CBoardState.
    It also contains some non-critic functions (whose performance is not critical) like displaying the board
    '''
    
    property debug:
        def __get__(self): return self.debug
        def __set__(self, bool b): self.debug = b
        
    property nRows:
        def __get__(self): return self.cBoardState.nRows
        
    property nCells:
        def __get__(self): return self.cBoardState.nCells
        
    property nPieces:
        def __get__(self): return self.cBoardState.nPieces
        
    property cells:
        def __get__(self): return self.cBoardState.getCells()
        
    
    def __cinit__(self, nRows=None, nPieces=None):
        self.debug = False
        if nPieces:
            self.cBoardState = new CBoardState(nRows, nPieces)
           
    def __dealloc__(self):
        del self.cBoardState
        
    def copy(self):
        copy = BoardState()
        copy.cBoardState = new CBoardState( deref(self.cBoardState) )
        return copy
        
    def reverse(self):
        self.cBoardState.reverse()
        return self      
        
    def __str__(self):
        return self.cBoardState.toString().decode('UTF-8')
         
    '''
    Checkers, converters, getters and setter
    '''       
    def isValidIndex(self, int i):
        return self.cBoardState.isValidIndex(i)
        
    def isValidRC(self, int r, int c):
        return self.cBoardState.isValidRC(r,c)
    
    def RCtoIndex(self, int r, int c):
        return self.cBoardState.RCtoIndex(r,c) 
        
    def indexToRC(self, int i):
        return self.cBoardState.indexToRC(i)
        
    def getCell(self, int r,int c):
        return self.cBoardState.getCell(r, c)
                
    def setCell(self, int i, cell):
        cdef char c = cell
        self.cBoardState.setCell(i, c)   
          
        
    ''' 
    Core methods 
    '''
             
    def tryMoveFrom(self, int cellIndex):
        cdef vector[CSimpleMove*] cmoves = self.cBoardState.tryMoveFrom(cellIndex)
        return [Move.wrap(m) for m in cmoves]    
    
    def tryJumpFrom(self, int cellIndex):
        cdef vector[CCaptureMove*] cmoves = self.cBoardState.tryJumpFrom(cellIndex)
        return [Move.wrap(m) for m in cmoves]
        
    def findPossibleMoves(self, bool white):
        cdef vector[CMove*] cmoves = self.cBoardState.findPossibleMoves(white)
        return [Move.wrap(m) for m in cmoves]
    
    def doMove(self, Move move):
        cdef CMove* c = move.cMove
        self.cBoardState.doMove( deref(c) )
        return self        
            
       
    def findNextStates(self, bool white):
        moves = self.findPossibleMoves(white)
        nextStates = []
        for m in moves:
            nextStates.append( self.copy().doMove(m) )
        return nextStates
        
 

    '''
    Visualization methods
    '''
    def toDisplay(self, showBoard = False):
        ''' Return a string suitable for state visualization in text mode (like the one at the top of this file) 
        If showBard is True, then a board with cell indices is shown next to the state'''
        
        formater = '{0:3d}'
        nRows = self.cBoardState.nRows
        
        s = ","+('---'*nRows)+","
        if showBoard:
            s+= "    ,"+('---'*nRows)+","
        s +="\n"
        for r in range(nRows):
            s+='|'
            for c in range(nRows):
                if c%2 != r%2:
                    s+=' '+str(chr(self.getCell(r,c)))+' '
                else:
                    s+='   '
            s+='|'
            if showBoard:
                s+='    |'
                for c in range(nRows):
                    if c%2 != r%2:
                        s+=formater.format(self.RCtoIndex(r,c))
                    else:
                        s+='   '
                s+='|'
            
            s+='\n'
        s+= "'"+('---'*nRows)+"'" 
        if showBoard:
            s+= "    '"+('---'*nRows)+"'"
        return s  
        
    def display(self, showBoard = False):
        print(self.toDisplay(showBoard))

        
    def viewBoard(self):
        formater = '{0:3d}'
        nRows = self.cBoardState.nRows
        
        s = ","+('---'*nRows)+",\n"
        for r in range(nRows):
            s+='|'
            for c in range(nRows):
                if c%2 != r%2:
                    s+=formater.format(self.RCtoIndex(r,c))
                else:
                    s+='   '
            s+='|\n'            
        s+= "'"+('---'*nRows)+"'\n"        
        return s
        
        

