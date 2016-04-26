#include <iostream>
#include <list>
#include "CBoardState.h"

namespace game {
    CBoardState::CBoardState() {}

    CBoardState::CBoardState(const int nR, const int nP) {
        nRows = nR;
        nPieces = nP;
        nCells = nRows*nRows/2;
        cells = std::vector<char>(nCells, '.');
        init();
    }
    
    CBoardState::CBoardState(const CBoardState& bs) { 
        nRows = bs.nRows;
        nPieces = bs.nPieces;
        nCells = bs.nCells;
        cells = std::vector<char>(bs.cells);     
    }

    CBoardState::~CBoardState() {}  
    
    void CBoardState::init(){   
        for (int i = 0; i < nPieces; i++)
            cells[i] = Cell::b;        
        for (int i = nPieces; i < nCells-nPieces; i++)
            cells[i] = Cell::empty;        
        for (int i = nCells-nPieces; i < nCells; i++)
            cells[i] = Cell::w;
    }

    std::vector<char> CBoardState::getCells(){
        return cells;
    }   
    
    void CBoardState::reverse(){
        char x;
        for (int i = 0; i < nCells/2; i++){
            x = cells[i];
            cells[i] = Cell::invertColor(cells[nCells-i-1]);
            cells[nCells-i-1] = Cell::invertColor(x); 
        }             
    } 
    
    std::string CBoardState::toString(){
        return std::string(cells.begin(), cells.end());
    }

/*
Convertissors, checkers and getters
*/
        
    bool CBoardState::isValidIndex(const int i){
        return (0<=i && i<nCells);
    }
    
    bool CBoardState::isValidRC(const int r, const int c){        
        return (0<=r && r<nRows) && (0<=c && c<nRows) && (r%2 != c%2);
    }
    
    int CBoardState::RCtoIndex(const int r, const int c){
        return r*(nRows/2) + c/2;
    } 
    
    std::pair<int,int> CBoardState::indexToRC(const int i){
        int r = i/(nRows/2);
        std::pair<int,int> rc (r, 2*(i%(nRows/2)) + (r+1)%2 );
        return rc;
    }   
    
    char CBoardState::getCell(const int i){
        if(not isValidIndex(i)){        
            std::cout << "Non valid index : " << i << "\n";
            throw "Non valid index";
        }
        return cells[i];
    }
        
    char CBoardState::getCell(const int r, const int c){
        if(!isValidRC(r,c)){        
            std::cout << "Non valid coordinates : " << r << ", " << c << "\n";
            throw "Non valid coordinates";
        }
        return cells[RCtoIndex(r,c)];
    }
        
    void CBoardState::setCell(const int i, const char c){
        if(!isValidIndex(i)) throw "Non valid coordinates";
        cells[i] = c;
    }
    
    
    
    
    
    
    std::vector<CCaptureMove*> CBoardState::tryJumpFrom(const int cellIndex){
        char piece = cells[cellIndex];
        if (piece == Cell::empty) throw "Error, the starting position contains no piece";   
        std::set<int> previousCaptures;
        std::vector<CCaptureMove*> possibleMoves = tryJumpFrom(cellIndex, cellIndex, piece, previousCaptures); 
        return possibleMoves;
    }
    
    std::vector<CCaptureMove*> CBoardState::tryJumpFrom(const int cellIndex, const int initPos, const char piece, std::set<int>& previousCaptures){
//        Find capturing moves that a iece located at a given position.
//        
//        Inputs:
//            - cellIndex : the starting position on the board.
//            - piece: the type of piece being moved
//        Output:
//            - moves: a list of valid variations. A variation is a list of jumps       
            
        std::pair<int,int> rc0 = indexToRC(cellIndex);
        int r0 = rc0.first;
        int c0 = rc0.second;          
        
        // trs stores valid row movements (downward for blacks, upward for whites, an both for kings)
        std::vector<int> trs, tcs;
        if (r0<nRows-2)  trs.push_back(1);
        if (r0>1)  trs.push_back(-1);
        if (c0<nRows-2)  tcs.push_back(1);
        if (c0>1)  tcs.push_back(-1);
        
        //a list of Move objects that the piece can perform starting from this place (these moves do not include the starting point)    
        std::vector<CCaptureMove*> possibleVariations; 
        std::vector<CCaptureMove*> newVariations;
        
        int r2,c2, jumpPos, newPos;  
        bool isWhite;
        char jumpedCell; 
        for (std::vector<int>::iterator itr = trs.begin(); itr != trs.end(); ++itr){
            for (std::vector<int>::iterator itc = tcs.begin(); itc != tcs.end(); ++itc){
                r2 = r0+2*(*itr);
                c2 = c0+2*(*itc);               
                jumpPos = RCtoIndex(r0+*itr,c0+*itc);
                newPos = RCtoIndex(r2,c2);               
                if (getCell(jumpPos)==Cell::empty || previousCaptures.find(jumpPos) != previousCaptures.end()) continue;
                
                isWhite = Cell::isWhite(piece);
                jumpedCell = getCell(jumpPos);             
                if ((getCell(newPos)==Cell::empty || newPos==initPos) && Cell::isWhite(jumpedCell) != isWhite){ 
                    // if this is a valid jump
                    if( !Cell::isKing(piece) && ((r2==0 && isWhite) || (r2==nRows-1 && !isWhite)) ){ 
                        // if a man has reached the last row, it has to stop and be crowned
                        possibleVariations.push_back( new CCaptureMove(newPos) );
                    }else{
                        std::set<int> newPreviousCaptures = std::set<int>(previousCaptures);
                        newPreviousCaptures.insert(jumpPos);
                        newVariations = tryJumpFrom(newPos, initPos, piece, newPreviousCaptures);                        
                        possibleVariations.insert(possibleVariations.end(), newVariations.begin(), newVariations.end());
                    }
                }
            }
        }
        
        
//        if (debug){
//            print(tabs,'possible variations before taking the max :')
//            for move in possibleVariations:print(tabs, move.toPDN())    
//            print() 
//        }      
        
        std::vector<CCaptureMove*> longestVariations;
        
        if(possibleVariations.size()>0){
            // Once we've gathered all possible variations from the four adjacent cells, we keep only the longest ones
            int currentMax = 0;
            int len;
            
            for (std::vector<CCaptureMove*>::iterator variation = possibleVariations.begin(); variation != possibleVariations.end(); ++variation){
                len = (*variation)->len();
                if (len > currentMax){
                    currentMax = len;
                    longestVariations.clear();
                }
                if (len >= currentMax){ 
                    //Then we insert the starting point to at the beginning of all variations kept  
                    (*variation)->push_front(cellIndex);       
                    longestVariations.push_back(*variation);
                }
            }
            
        }else if(!previousCaptures.empty()){
            // if there is no move possible from this cellIndex and that there has been previous capture,
            // then it means cellIndex is the end of a move and we need to put it in the possible moves
            longestVariations.push_back( new CCaptureMove(cellIndex) );        
        }
        
//        if self.debug:
//            print(tabs,'possible ends of the move :')
//            for move in moves:print(tabs, move.toPDN())    
//            print() 
          
        return longestVariations;
    }
    

    std::vector<CSimpleMove*> CBoardState::tryMoveFrom(const int cellIndex){
        char piece = cells[cellIndex];
        if (piece==Cell::empty){
            std::cout << "Error, the starting position contains no piece";
            throw "Error, the starting position contains no piece";
        }
        
        std::pair<int,int> rc = indexToRC(cellIndex);
        int r = rc.first;
        int c = rc.second;
        
        // trs stores valid row movements (downward for blacks, upward for whites, an both for kings)
        std::vector<int> trs, tcs;
        if (r<nRows-1 && piece!=Cell::w)  trs.push_back(1);
        if (r>0  && piece!=Cell::b)  trs.push_back(-1);
        if (c<nRows-1)  tcs.push_back(1);
        if (c>0)  tcs.push_back(-1);
             
        std::vector<CSimpleMove*> possibleMoves;
        int newPos;
        for (std::vector<int>::iterator itr = trs.begin(); itr != trs.end(); ++itr){
            for (std::vector<int>::iterator itc = tcs.begin(); itc != tcs.end(); ++itc){
                newPos = RCtoIndex(r+*itr,c+*itc);                       
                if (cells[newPos]==Cell::empty) possibleMoves.push_back( new CSimpleMove(cellIndex, newPos) );
            }
        }       
                       
        return possibleMoves;
    }
    
    
    std::vector<CMove*> CBoardState::findPossibleMoves(const bool white){
   //     Find valid moves and their corresponding states for a given player.
   //     
   //     Input: 
   //         - color: the color of the player we want to find moves
   //     Outputs:
   //         - moves: the list containing very authhorized move.  
               
        std::vector<CMove*> moves;
        
        // First we look for capturing moves (we are obliged to capture as many pieces as possible)
        std::vector<CCaptureMove*> pieceMoves;
        int maxCaptures = 0;
        int nbCaptures;
        char piece;
        for(int i = 0; i<nCells; ++i){
            piece = cells[i];
            if(piece!=Cell::empty && Cell::isWhite(piece)==white){
                pieceMoves = tryJumpFrom(i);            
                if(!pieceMoves.empty()){
                    nbCaptures = pieceMoves[0]->len();
                    if(nbCaptures > maxCaptures){
                        moves.clear();
                        maxCaptures = nbCaptures;
                    }
                    if(nbCaptures >= maxCaptures){
                        moves.insert(moves.end(), pieceMoves.begin(), pieceMoves.end());
                    }
                }
            }   
        }                    
        
        // If there is no capture move possible, we find simple moves                   
        if(moves.empty()){
            std::vector<CSimpleMove*> pieceMoves;
            for(int i = 0; i<nCells; ++i){
                piece = cells[i];
                if(piece!=Cell::empty && Cell::isWhite(piece)==white){
                    pieceMoves = tryMoveFrom(i);
                    moves.insert(moves.end(), pieceMoves.begin(), pieceMoves.end());
                }
            }
        }
        
        return moves;
    }
    
    
    void CBoardState::doMove(const CMove& move){
        /* Update the state according to the specified move 
        
        Note that this function does not check if the move is valid*/
        std::list<int> mCells = move.getCells();
        int start = mCells.front();
        if (!isValidIndex(start)) {throw "Invalid Index !";}
        int end = mCells.back();        
        char piece = cells[start];
        int diff1,diff2;
        
        if(move.isCapture()){
            std::pair<int,int> RC = indexToRC(start);
            std::pair<int,int> nextRC;
            for (std::list<int>::iterator it = ++mCells.begin(); it != mCells.end(); ++it){
                if (!isValidIndex(*it)) {throw "Invalid Index !";}
                nextRC = indexToRC(*it);
                
                diff1 = RC.first-nextRC.first;
                diff2 = RC.second-nextRC.second;
                if(!( (diff1==2 || diff1==-2) && (diff2==2 || diff2==-2) )) {throw "Invalid Move !";}
                
                cells[RCtoIndex( (RC.first+nextRC.first)/2, (RC.second+nextRC.second)/2 )] = Cell::empty;
                RC = nextRC;
            }
        }
        cells[start] = Cell::empty;
        cells[end] = piece;  
        std::pair<int,int> endRC = indexToRC(end);
                
        if( (piece == Cell::w && endRC.first==0)  || (piece == Cell::b && endRC.first==nRows-1) ){
            cells[end] = Cell::promote(piece);
        }
    }
     
}
