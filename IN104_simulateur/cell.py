class Cell():
    ''' This enumeration represents the possible states of a cell on the board '''
    empty = 46
    w = 119
    W = 87
    b = 98
    B = 66  
                            
    def isWhite(c):
        return c == Cell.w or c == Cell.W          
        
    def isBlack(c):
        return c == Cell.b or c == Cell.B     
      
    def isMan(c):
        return c == Cell.w or c == Cell.b         
        
    def isKing(c):
        return c == Cell.W or c == Cell.B        
    
    def invertColor(c):
        if c == Cell.empty: return Cell.empty 
        return v + 42*(1-v%2) - 21
        
    def promoted(c):
        assert(Cell.isMan(c))
        return c-32
        
    def toString(c):
        return chr(c)   
