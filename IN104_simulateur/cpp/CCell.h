namespace game { 
    namespace Cell {
        const char empty = '.';
        const char w = 'w';
        const char W = 'W';
        const char b = 'b';
        const char B = 'B';
        
        bool isWhite(const char p);
        bool isBlack(const char p);
        bool isMan(const char p);
        bool isKing(const char p);
        char invertColor(const char p);
        char promote(const char p);
    }  
 }
