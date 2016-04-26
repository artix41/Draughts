#include "CMove.h"

namespace game{
      
    
    std::list<int> CMove::getCells() const{
        return cells;
    }
    
    int CMove::len() const{
        return cells.size();
    }
    
    bool CMove::isCapture() const{
        return isCap;
    }
        
    std::string CMove::toPDN(){
        std::ostringstream ss;
        ss << cells.front();
        for(std::list<int>::iterator it = ++cells.begin(); it != cells.end(); ++it){
            ss << separator << *it;
        }
        return ss.str();
    }
    
    
    
    CSimpleMove::CSimpleMove(){}  
            
    CSimpleMove::CSimpleMove(const int cell1, const int cell2){
        cells.push_back(cell1);
        cells.push_back(cell2);
        separator = "-";
        isCap = false;
    }
    
    
    CCaptureMove::CCaptureMove(){} 
            
    CCaptureMove::CCaptureMove(const int cell){
        cells.push_front(cell);
        separator = "x";
        isCap = true;
    }
    
    void CCaptureMove::push_front(const int cell){
        cells.push_front(cell);
    }
    
    void CCaptureMove::push_back(const int cell){
        cells.push_back(cell);
    }
 }
