from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list

cdef extern from "CMove.h" namespace "game":
    cdef cppclass CMove:  
        list[int] getCells()
        int len()  
        bool isCapture()
        string toPDN()
        
    cdef cppclass CCaptureMove(CMove):
        CCaptureMove() except +   
        CCaptureMove(int) except +
        void push_back(int) 
        
    cdef cppclass CSimpleMove(CMove):
        CSimpleMove() except +  
        CSimpleMove(int, int) except + 


cdef class Move:    
    cdef CMove* cMove
    
    @staticmethod
    cdef wrap(CMove* m)
    
