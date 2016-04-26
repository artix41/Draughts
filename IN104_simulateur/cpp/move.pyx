# distutils: language = c++
# distutils: sources = IN104_simulateur/cpp/CBoardState.cpp IN104_simulateur/cpp/CCell.cpp IN104_simulateur/cpp/CMove.cpp
from cython.operator cimport dereference as deref, preincrement as inc

from .move cimport *

cdef class Move:
    ''' A move is simply a list of cells which a piece passes by. 
    
    It can be either a CaptureMove or a SimpleMove. 
    These classes do not check whether the list of cells define a valid move'''
        
    def __cinit__(self):
        pass     
        
    def __dealloc__(self):
        del self.cMove
    
    @staticmethod
    cdef wrap(CMove* m):
        result = Move()
        result.cMove = m
        return result         
 
    def __len__(self):
        return self.cMove.len()       

    
    def isCapture(self):
        return self.cMove.isCapture()
    
    def toPDN(self):
        return self.cMove.toPDN().decode('UTF-8')
        
    @staticmethod
    def fromPDN(s):
        assert(s.rfind('x')>-1 or s.rfind('-')>-1)
        separator = 'x' if s.rfind('x')>-1 else '-'
        cdef vector[int] cells = [int(i) for i in s.split(separator)]
        assert(separator=='x' and len(cells)>=2 or len(cells)==2)
        
        move = Move()   
        cdef CSimpleMove* csmove
        cdef CCaptureMove* ccmove
        if separator=='-':
            csmove = new CSimpleMove(cells[0], cells[1])
            move.cMove = csmove
        else:
            ccmove = new CCaptureMove(cells[0])
            for k in range(1,len(cells)):
                ccmove.push_back(cells[k])
            move.cMove = ccmove
             
        return move
