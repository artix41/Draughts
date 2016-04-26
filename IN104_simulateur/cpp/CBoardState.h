#pragma once 
//#include <memory> // include std::unique_ptr
#include <vector>
#include <string>
#include <utility> // include std::pair
#include <set>
#include "CCell.h"
#include "CMove.h"

namespace game {
    class CBoardState {       
    private:        
        std::vector<char> cells;  
    public:
        int nRows, nCells, nPieces;
                
        CBoardState();
        CBoardState(const int nRows, const int nPieces);
        CBoardState(const CBoardState& bs);
        ~CBoardState();    
        
        void init();
        std::vector<char> getCells();
        void reverse();
        std::string toString();
        
        bool isValidIndex(const int i);
        bool isValidRC(const int r, const int c);
        int RCtoIndex(const int r, const int c);
        std::pair<int,int> indexToRC(const int i);    
        char getCell(const int i);
        char getCell(const int r, const int c);
        void setCell(const int i, const char c);
        
        std::vector<CCaptureMove*> tryJumpFrom(const int cellIndex);
        std::vector<CCaptureMove*> tryJumpFrom(const int cellIndex, const int initPos, const char piece, std::set<int>& previousCaptures);
        std::vector<CSimpleMove*> tryMoveFrom(const int cellIndex);
        std::vector<CMove*> findPossibleMoves(const bool white);
        void doMove(const CMove& move);
        //std::vector<CMove> findPossibleMoves(const bool color);           
    };
 }
