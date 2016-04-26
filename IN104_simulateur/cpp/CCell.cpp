#include "CCell.h"
#include <cstddef>
#include <iostream>

namespace game{
    namespace Cell{
        bool isWhite(const char p){
            return p==Cell::w or p==Cell::W;
        }        
        
        bool isBlack(const char p){
            return p==Cell::b or p==Cell::B;
        }
        
        
        bool isMan(const char p){
            return p==Cell::b or p==Cell::w;
        }
        
        bool isKing(const char p){
            return p==Cell::B or p==Cell::W;
        }
        
        char invertColor(const char p){
            if (p==Cell::empty) return p;
            if (isWhite(p)) return p-21;
            if (isBlack(p)) return p+21;
            throw "invalid cell";
        }
        
        char promote(const char p){
            return p-32;
        } 
    }
}
