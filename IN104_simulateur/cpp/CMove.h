#pragma once 
#include <list>
#include <sstream>
#include <string>

namespace game{
    class CMove{
    protected:
        bool isCap;
        std::string separator;
        std::list<int> cells;
    public:      
        int len() const;
        std::list<int> getCells() const;
        bool isCapture() const; 
        std::string toPDN(); 
    };
    
    class CSimpleMove: public CMove{
    public:
        CSimpleMove();
        CSimpleMove(const int cell1, const int cell2);
    };
    
    class CCaptureMove: public CMove{
    public:
        CCaptureMove();
        CCaptureMove(int cell);
        void push_front(const int cell);
        void push_back(const int cell);
    };

 }
