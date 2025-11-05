# CMPUT 455 Assignment 3 starter code (PoE2)
# Implement the specified commands to complete the assignment
# Full assignment specification on Canvas
"""
todo:
    [] undo exception for when the are undoing moves that were initialized, but we don't know what previous move was so shouldn't be able to do
    [] time outs caused by empty boards and lots of patterns
    """


import sys
import signal
import copy

class CommandInterface:
    # The following is already defined and does not need modification
    # However, you may change or add to this code as you see fit, e.g. adding class variables to init

    def __init__(self):
        # Define the string to function command mapping
    
        self.command_dict = {
            "help"     : self.help,
            "init_game": self.init_game,   # init_game w h p s [board]
            "show"     : self.show,
            "timelimit": self.timelimit,   # timelimit seconds
            "load_patterns"    : self.load_patterns,       # see assignment spec
            "policy_moves"    : self.policy_moves,       # see assignment spec
            "position_evaluation": self.position_evaluation, # see assignment spec
            "move_evaluation": self.move_evaluation, # see assignment spec
            "score"    : self.score,
            "play" : self.play
        }

        # Game state
        self.board = [[None]]
        self.player = 1           # 1 or 2
        self.handicap = 0.0       # P2’s handicap
        self.score_cutoff = float("inf")

        

        # This variable keeps track of the maximum allowed time to solve a position
        # default value
        #self.timelimit = 1
        self.timelimit = 10000
        
        self.patterns = [] # you may change this
        self.patternVal = {}


    # Convert a raw string to a command and a list of arguments
    def process_command(self, s):
        
        class TimeoutException(Exception):
            pass
        
        def handler(signum, frame):
            raise TimeoutException("Function timed out.")
        
        s = s.lower().strip()
        if len(s) == 0:
            return True
        command = s.split(" ")[0]
        args = [x for x in s.split(" ")[1:] if len(x) > 0]
        if command not in self.command_dict:
            print("? Uknown command.\nType 'help' to list known commands.", file=sys.stderr)
            print("= -1\n")
            return False
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.timelimit)
            
            return self.command_dict[command](args)
        except TimeoutException:
            print(f"Command '{s}' timed out after {self.timelimit} seconds.", file=sys.stderr)
            print("= -1\n")
            return False
        except Exception as e:
            print("Command '" + s + "' failed with exception:", file=sys.stderr)
            print(e, file=sys.stderr)
            print("= -1\n")
            return False
        finally: 
            signal.alarm(0)

        
    # Will continuously receive and execute commands
    # Commands should return True on success, and False on failure
    # Every command will print '= 1' or '= -1' at the end of execution to indicate success or failure respectively
    def main_loop(self):
        while True:
            s = input()
            if s.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(s):
                print("= 1\n")

    # List available commands
    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                print(command)
        print("exit")
        return True

    # Helper function for command argument checking
    # Will make sure there are enough arguments, and that they are valid integers
    def arg_check(self, args, template):
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr)
            print("Recieved arguments: ", end="", file=sys.stderr)
            for a in args:
                print(a, end=" ", file=sys.stderr)
            print(file=sys.stderr)
            return False
        for i, arg in enumerate(args):
            try:
                args[i] = int(arg)
            except ValueError:
                try:
                    args[i] = float(arg)
                except ValueError:
                    print("Argument '" + arg + "' cannot be interpreted as a number.\nExpected arguments:", template, file=sys.stderr)
                    return False
        return True


    # init_game w h p s [board_str]
    # Note that your init_game function must support initializing the game with a string (this was not necessary in A1).
    # We already have implemented this functionality in our provided init_game function.
    def init_game(self, args):
        # Check arguments
        if len(args) > 4:
            self.board_str = args.pop()
        else:
            self.board_str = ""
        if not self.arg_check(args, "w h p s"):
            return False
        w, h, p, s = args
        if not (1 <= w <= 10000 and 1 <= h <= 10000):
            print("Invalid board size:", w, h, file=sys.stderr)
            return False
        


        
        #Initialize game state
        self.width = w
        self.height = h
        self.handicap = p
        self.patternMatches = []
        self._moveHistory = [] #((row:int,col:int),(_p1Score:float, _p2Score:float), cur_player:int, winner:int)
        if s == 0:
            self.score_cutoff = float("inf")
        else:
            self.score_cutoff = s
        
        self.board = []
        for r in range(self.height):
            self.board.append([0]*self.width)
        self.player = 1

        # optional board string to initialize the game state
        if len(self.board_str) > 0:
            board_rows = self.board_str.split("/")
            if len(board_rows) != self.height:
                print("Board string has wrong height.", file=sys.stderr)
                return False
            
            p1_count = 0
            p2_count = 0
            for y, row_str in enumerate(board_rows):
                if len(row_str) != self.width:
                    print("Board string has wrong width.", file=sys.stderr)
                    return False
                for x, c in enumerate(row_str):
                    if c == "1":
                        self.board[y][x] = 1
                        p1_count += 1
                        # add this move to the list of moves played order is arbitrary and ambiguous
                        # how does this impact undo, and will undo ever be called?
                        self._moveHistory.append(((y,x),(-1, -1), 1 , -1))
                    elif c == "2":
                        self.board[y][x] = 2
                        p2_count += 1
                        # add this move to the history, order is ambiguous and arbitrary
                        self._moveHistory.append(((y,x),(-1, -1), 2 , -1))

            self._numMoves = p1_count + p2_count
            if p1_count > p2_count:
                self.player = 2
            else:
                self.player = 1

        #defualt time limit
        # self.timelimit = 1
        self.timelimit = 10000

        # Game state  
        # all caps = const
        self._SCORECUTOFF = float(s)
        self._NUMTILES = self.width * self.height
        self._HANDICAP = self.handicap
        self._won = 0
        self._numMoves = 0
        self.player= self.player # 1 _player 1 or 2 _player 2
        self._p1Score = int()
        self._p2Score = float(p)
        self._newGame = True ## could change this whole check to just be if the board is empty, but this would be called often, and so it would be slow to compute

        self._p1Score, self._p2Score = self.calculate_scoreFull()
        self.is_terminalFast(self._p1Score, self._p2Score)

        return True

    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True

    def timelimit(self, args):
        """
        >> timelimit <seconds>
        Sets the wall-clock time limit used by 'solve'.
        - Accepts a single non-negative integer.
        """
        if not self.arg_check(args, "s"):
            return False

        self.timelimit = int(args[0])
        return True
    
    def timeLimit(self, args):
        """
        this is for me because it has error to call int object
        """
        if not self.arg_check(args, "s"):
            return False

        self.timelimit = int(args[0])
        return True
    
    # The following functions do not need to be callable as commands in assignment 2, but implement the PoE2 game environment for you.
    # Feel free to change or modify or replace as needed, your implementation of A1 may provide better optimized methods.
    # These functions work, but are not necessarily computationally efficient.
    # There are different approaches to exploring state spaces, this starter code provides one approach, but you are not required to use these functions.

    def get_moves(self):
        moves = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    moves.append((x, y))
        return moves

    def make_move(self, x, y):
        self.board[y][x] = self.player
        self._moveHistory.append(((y,x),(-1, -1), self.player , -1))
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def undo_moveSimple(self, x, y):
        """
        Q! there is no command that says undo, so it is not able to be tested
        """
        self.board[y][x] = 0
        self._moveHistory.pop()
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1


    def undo_move(self):
        '''
        Undoes the last move.

        todo: 
            [ ] change it so that board initialization is handled

        Returns:
            True if undo was successful 

            False if no moves to undo
        '''
        if not self._moveHistory:
            self._newGame = True
            return False
        
        status = self._moveHistory.pop()
        self._numMoves -= 1
        self.board[status[0][0]][status[0][1]] = 0
        if self._moveHistory:
            newState = self._moveHistory[-1]
            self._p1Score = newState[1][0]
            self._p2Score = newState[1][1]
            self.player= newState[2]
            self._won = newState[3]
        else:
            self._numMoves = 0
            self._p1Score = 0
            self._p2Score = self.HANDICAP
            self.player= 1
            self._won = 0
        return True

    # Returns p1_score, p2_score
    def calculate_scoreFull(self):
        """
        calculate the score from scratch
        """
        p1_score = 0
        p2_score = self.handicap

        # Progress from left-to-right, top-to-bottom
        # We define lines to start at the topmost (and for horizontal lines leftmost) point of that line
        # At each point, score the lines which start at that point
        # By only scoring the starting points of lines, we never score line subsets
        for y in range(self.height):
            for x in range(self.width):
                c = self.board[y][x]
                if c != 0:
                    lone_piece = True # Keep track of the special case of a lone piece
                    # Horizontal
                    hl = 1
                    if x == 0 or self.board[y][x-1] != c: #Check if this is the start of a horizontal line
                        x1 = x+1
                        while x1 < self.width and self.board[y][x1] == c: #Count to the end
                            hl += 1
                            x1 += 1
                    else:
                        lone_piece = False
                    # Vertical
                    vl = 1
                    if y == 0 or self.board[y-1][x] != c: #Check if this is the start of a vertical line
                        y1 = y+1
                        while y1 < self.height and self.board[y1][x] == c: #Count to the end
                            vl += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Diagonal
                    dl = 1
                    if y == 0 or x == 0 or self.board[y-1][x-1] != c: #Check if this is the start of a diagonal line
                        x1 = x+1
                        y1 = y+1
                        while x1 < self.width and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            dl += 1
                            x1 += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Anit-diagonal
                    al = 1
                    if y == 0 or x == self.width-1 or self.board[y-1][x+1] != c: #Check if this is the start of an anti-diagonal line
                        x1 = x-1
                        y1 = y+1
                        while x1 >= 0 and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            al += 1
                            x1 -= 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Add scores for found lines
                    for line_length in [hl, vl, dl, al]:
                        if line_length > 1:
                            if c == 1:
                                p1_score += 2 ** (line_length-1)
                            else:
                                p2_score += 2 ** (line_length-1)
                    # If all found lines are length 1, check if it is the special case of a lone piece
                    if hl == vl == dl == al == 1 and lone_piece:
                        if c == 1:
                            p1_score += 1
                        else:
                            p2_score += 1

        return p1_score, p2_score
    
    def score(self, args):
        print(self._p1Score, self._p2Score)
        # print(self.calculate_score())
    
    # Returns is_terminal, winner
    # Assumes no draws
    def is_terminalSlow(self):
        """
        this may be faster with a linear score calculation, if score calc is faster than going throught the full board
        but then just consider my other implementation that counts the number of tiles played
        """
        p1_score, p2_score = self.calculate_score()
        if p1_score >= self.score_cutoff:
            return True, 1
        elif p2_score >= self.score_cutoff:
            return True, 2
        else:
            # Check if the board is full
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] == 0:
                        return False, 0
            # The board is full, assign the win to the greater scoring player
            if p1_score > p2_score:
                return True, 1
            else:
                return True, 2


    def is_terminalFast(self, p1_score=-1, p2_score=-1): 
        """
        faster considering slow implementation of calc score
        this changes order to minimize calls to calculate_scoreFull
        """

        # Check if the board is full
        full = True
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    full = False
                    break
        
        if self.score_cutoff == float("inf") and not full:
            return False, 0
        
        if p1_score == -1 or p2_score == -1:
            p1_score, p2_score = self.calculate_scoreFull() # this is what I need to change

        if full:
            if p1_score > p2_score:
                return True, 1
            else:
                return True, 2
            
        if p1_score >= self.score_cutoff:
            return True, 1
        elif p2_score >= self.score_cutoff:
            return True, 2
        else:
        # The board is full, assign the win to the greater scoring player
            return False, 0
    
    def is_terminal(self):
        """
        using my game implementations
        """
        if self._SCORECUTOFF == 0 and self._numMoves != self._NUMTILES:
            self._won = 0
            return False

        elif self._SCORECUTOFF != 0 and self._p1Score >= self._SCORECUTOFF:
            self._won = 1
            return True
        
        elif self._SCORECUTOFF != 0 and self._p2Score >= self._SCORECUTOFF:
            self._won = 2
            return True
        
        elif self._numMoves == self._NUMTILES:
            if self._p1Score > self._p2Score:
                self._won = 1
            else:
                self._won = 2 
            return True
        
        else:
            self._won = 0

        assert self._numMoves <= self._NUMTILES, "number of tiles exceed possible"
        return False, self._won
    
    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True
            
    
    def isLegal(self, c, r):
        if self._won == 0 and self.board[r][c] == 0:
            return True
        
        return False      
    
    # this function may be modified as needed, but should behave as expected
    def play(self, args):
        '''
            >> play <col> <row>
            Places current player's piece at position (<col>, <row>).
        '''
        row = int(args[1])
        col = int(args[0])
        if not self.isLegal(col, row):
            return -1
        if self._won != 0:
            return self._won

        ## could these be assertions, or then this would stop the program which is not what I want
        # It is not a newGame since a move has been played
        self._newGame = False
        # update the board state
        
        # calculate the score through all the four possible directions of lines
        self.calculate_score(row, col)

        # keep track of the number of moves played
        self._numMoves += 1 

        self.is_terminal()

        # add the move history to the list for undo
        self._moveHistory.append(((row,col),(self._p1Score, self._p2Score), self.player, self._won))

        # update the board needs to happen after calcScore because it needs to check for neighbors that are not the new move
        self.board[row][col] = self.player
        if self.player== 2:
            self.player= 1
        else:
            self.player= 2
    

        return 1
    
        # this confuses me it never actually calls make move and is very slow
        # if not self.arg_check(args, "x y"):
        #     return False
        
        # try:
        #     col = int(args[0])
        #     row = int(args[1])
        # except ValueError:
        #     #print("Illegal move: " + " ".join(args), file=sys.stderr)
        #     return False
        
        # if col < 0 or col >= len(self.board[0]) or row < 0 or row >= len(self.board) or not self.is_pos_avail(col, row):
        #     raise Exception("Illegal Move")
            
        
        # scores = self.calculate_score()
        # if scores[0] >= self.score_cutoff or scores[1] >= self.score_cutoff:
        #     #print("Illegal move: " + " ".join(args), "game ended.", file=sys.stderr)
        #     return False

        # # put the piece onto the board
        # self.board[row][col] = self.player

        # # compute the score for both players after each round
        # self.calculate_score()

        # # record the move
        # # self.move_history.append((col, row, self.player))

        # # switch player
        # if self.player == 1:
        #     self.player = 2
        # else:
        #     self.player = 1

        
        # return True
    
    
    def calculate_score(self, row:int, col:int)->int:
        """
        adds the score for all types of lines that can be made Linearly
        """
        # assert proper row and col is given
        assert 0 <= row < self.height, "row is out of bounds"
        assert 0 <= col < self.width, "col is out of bounds"

        flag = 0
        # check vertical neighbors
        if self._lineCheck(row, col, 1, 0):
            flag = 1
        # check horizontal neighbors
        if self._lineCheck(row, col, 0, 1):
            flag = 1
        # check the diagonal neighbors
        if self._lineCheck(row, col, 1, 1):
            flag = 1
        # check the antidiagonal
        if self._lineCheck(row, col, -1, 1):
            flag = 1
        if flag == 0:
            if self.player== 1:
                self._p1Score += 1
            else:
                self._p2Score += 1

        return 

    def _lineCheck(self, row:int, col:int, rowShift:int, colShift:int)-> int:
        """
        rowshift is if checking the most south or east tile must shift the row down

        colshift is if checking the most south or east tile must shift the col down

        used to add the scores for a row and column of a given line type (dia, antidia, vertical, horizontal)
        
        Formula: + new line length points - old points contributed by num above - old points contributed by num below
        """
        a = (1*rowShift)
        b = (-1*rowShift)
        c = (1*colShift)
        d = (-1*colShift)


        posRow = row + a
        negRow = row + b
        posCol = col + c
        negCol = col + d

        numAbove = 0
        numBelow = 0

        while 0 <= negRow < self.height and 0 <= negCol < self.width and self.board[negRow][negCol] == self.player:
            numBelow += 1
            negRow += b
            negCol += d
        while 0 <= posRow < self.height and 0 <= posCol < self.width and self.board[posRow][posCol] == self.player:
            numAbove += 1
            posRow += a
            posCol += c

        if numAbove == 0 and numBelow == 0:
            return False
        
        # update the score
        if self.player== 1:
            self._p1Score += 2**(numAbove + numBelow)
        else:
            self._p2Score += 2**(numAbove + numBelow)

        # need to find the supersets to remove
        # if above does not have a neighbor, subtract 1 because it added 1 before
        #* if above has a neighbor and the num above == 1 then don't subtract any becaus it's line of that type did not exist / this is the main thing that I was missing for it to work
        # else subtract numAbove-1
        ## I feel I am missing something for double connctions but maybe not because I add the whole line then get rid of the rest
        posRow = row + a
        negRow = row + b
        posCol = col + c
        negCol = col + d

        if numAbove != 0:
            # this should already hold and may be redundant
            assert 0 <= posRow < self.height and 0 <= posCol < self.width, "checking for an above cell that is out of bounds"
            neigh = self.__checkIfNeighbor(posRow, posCol)
            if self.player== 1:
                if not neigh:
                    self._p1Score -= 1
                elif numAbove != 1:
                    self._p1Score -= 2**(numAbove-1)
            else:
                if not neigh:
                    self._p2Score -= 1
                elif numAbove != 1:
                    self._p2Score -= 2**(numAbove-1)

        if numBelow != 0:
            # this should already hold and may be redundant
            assert 0 <= negRow < self.height and 0 <= negCol < self.width, "checking for a below cell that is out of bounds"
            neigh = self.__checkIfNeighbor(negRow, negCol)
            if self.player== 1:
                if not neigh:
                    self._p1Score -= 1
                elif numBelow != 1:
                    self._p1Score -= 2**(numBelow-1)
            else:
                if not neigh:
                    self._p2Score -= 1
                elif numBelow != 1:
                    self._p2Score -= 2**(numBelow-1)

        return True
    

    def __checkIfNeighbor(self, row:int, col:int)->int:
        """
        used to check if a given tile has any neighbors
        """
        # assert proper row and col is given
        assert 0 <= row < self.height, "row is out of bounds, neigh"
        assert 0 <= col < self.width, "col is out of bounds, neigh"

        # check antidia, diagonal, horizontal, vertical
        return self.__checkSuperset(row, col, -1, 1) or self.__checkSuperset(row, col, 1, 1) or self.__checkSuperset(row, col, 0, 1) or self.__checkSuperset(row, col, 1, 0)

    def __checkSuperset(self, row:int, col:int, rowShift:int, colShift:int)-> int:
        """
        rowshift is if checking the most south or east tile must shift the row down
        colshift is if checking the most south or east tile must shift the col down
        
        returns:
            True if there is a neighbor that is of the same type
            False if no neighbor
        """
        posRow = row + (1*rowShift)
        negRow = row + (-1*rowShift)
        posCol = col + (1*colShift)
        negCol = col + (-1*colShift)

        if 0 <= negRow < self.height and 0 <= negCol < self.width:
            if self.board[negRow][negCol] == self.player:
                return True

        if 0 <= posRow < self.height and 0 <= posCol < self.width:
            ##s can update it so that there is priority for each players move or only your moves count
            if self.board[posRow][posCol] == self.player:
                return True

        return False



    
    # new function to be implemented for assignment 3
    def load_patterns(self, args):
        with open(args[0], 'r') as f:
            # reset the array of patterns
            self.patterns = []
            self.patternVal = {}
            for line in f:
                # add the line split by the space to a tuple, do I want it as a tuple or list
                pat, val = line.split(" ")

                # eliminate any doubles on the end, since they don't add any context
                while pat.find("XX") != -1:
                    pat = pat.replace("XX", "X")

                # val is a float so we can do arithmetic pat is a string
                if (pat, float(val)) not in self.patterns:
                    self.patterns.append((pat, float(val)))
        
        # Sort by string length (descending), then by float value (descending) as tiebreaker
        self.patterns = sorted(self.patterns, key=lambda x: (len(x[0]), x[1]), reverse=True)
        return True
    
    # new function to be implemented for assignment 3
    def policy_moves(self, args):
        policy = []
        movesEval = self.move_evaluation((696969))
        total = sum(movesEval)
        low = min(movesEval)
        reduce = (-1*low+1)
        dom = total + len(movesEval)*reduce
        for ev in movesEval:
            policy.append(round((ev + reduce)/dom, 3))

        print(*policy)
        return policy

    # new function to be implemented for assignment 3
    def move_evaluation(self, args):
        """
        Q! 
            - does it matter if the position is terminal or not?
        """
        # check for empty moves, faster than going through it twice with get_moves
        moveVal = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    self.make_move(x, y)
                    moveVal.append(-1*self.position_evaluation(696969))
                    self.undo_moveSimple(x, y)
        
        moves  = [round(x, 1) if round(x, 1) != 0 else 0 for x in moveVal]
        if args != 696969:
            print(*moves)
        return moveVal


    # new function to be implemented for assignment 3
    def position_evaluation(self, args)-> float:
        """
        todo:
            [x] consider subsets using the fact we look at the longest first
            [x] consider if the length pushs the rotation of of bounds when it is not supposed to be 
            [] if already found it for same pattern than return right away
                - idk if this is possible
            [] consider both forwards and backwards at same time
                - kinda pointless to do at same time


        todo:**
            [x] consider diagonal lines that stem from the perimeter

        """
        # new position to evaluate so reset parameters
        self.patternMatches = []
        self.value = 0

        # get opp
        if self.player == 2:
            self.opp = 1
        else:
            self.opp = 2
        # get the pattern
        for self.pattern, self.patVal in self.patterns:
            # find the first instance of the bestType in pattern
            bestType = self.findBestType(self.pattern)
            # if walls go through all wall versions then call the rotations
            if bestType == "X":
                """
                need to consider X* since * can be another wall so the pruning does not help, really the only way is that I have to compare it multiple times, so just create all combinations
                """
                # everything is of the form X****
                # compare all verticals going down
                #! could save time by doing some check for length being cut off so only consider long enough diagonals

                for col in range(self.width):
                    # straight and 2 diagonals
                    self.matchLine((1,0), 0, col, 1, 1)
                    self.matchLine((1,1), 0, col, 1, 1)
                    self.matchLine((-1,1), 0, col, 1, 1)
                    # far right
                    self.matchLine((1,0), self.height-1, col, 1, -1)
                    self.matchLine((1,1), self.height-1, col, 1, -1)
                    self.matchLine((-1,1), self.height-1, col, 1, -1)
                # compare all horizontals going left
                for row in range(self.height):
                    # top
                    self.matchLine((0,1), row, 0, 1, 1)
                    self.matchLine((1,1), row, 0, 1, 1)
                    self.matchLine((-1,1), row, 0, 1, 1)
                    # bottom
                    self.matchLine((0,1), row, self.width-1, 1, -1)
                    self.matchLine((1,1), row, self.width-1, 1, -1)
                    self.matchLine((-1,1), row, self.width-1, 1, -1)

                # diagonals
                    self.matchLine((1,1), -1, -1, 0, 1)
                    self.matchLine((1,1), self.height, self.width, 0, -1)
                # antidiagonals
                    self.matchLine((-1,1), self.height, -1, 0, 1)
                    self.matchLine((-1,1), -1, self.width, 0, -1)
                print("x")
            # if player go through player history then call the rotations, have O first because likely to have less of p1
            elif bestType == "O":
                # get a move that needs to be checked ie a cell with an opponents move in it
                for move in self._moveHistory:
                    if move[2] == self.opp:
                        self.matchPattern(move[0], self.pattern.find("O"))
                print("O")
                        
                    
            elif bestType == "P":
                for move in self._moveHistory:
                    if move[2] == self.player:
                        self.matchPattern(move[0], self.pattern.find("P"))
                print("p")
            # if empty go through entire board to find it then call the rotations
            elif bestType == "_":
                for y in range(self.height):
                    for x in range(self.width):
                        if self.board[y][x] == 0:
                            self.matchPattern((y, x), self.pattern.find("_"))
                print("_")
                
        if args != 696969:
            print(self.value)
        return float(self.value)

    def matchPattern(self, pos:tuple, index):
        """
        this function finds the instances of the best type and then calls for all 4 line types and their symmetries
        args:
            pos: (row, col)
            index: (where in the pattern is the first occurence of the type)
        """
        # get pos vertical
        self.matchLine((1,0), pos[0]-index, pos[1], 1)
        # get neg vertical
        self.matchLine((1,0), pos[0]+index, pos[1], -1)
        # get pos horizontal
        self.matchLine((0,1), pos[0], pos[1]-index, 1)
        # get neg horizontal
        self.matchLine((0,1), pos[0], pos[1]+index, -1)
        # get pos diagonal
        self.matchLine((1,1), pos[0]-index, pos[1]-index, 1)
        # get neg diagonal
        self.matchLine((1,1), pos[0]+index, pos[1]+index, -1)
        # get pos antidiagonal
        self.matchLine((-1,1), pos[0]+index, pos[1]-index, 1)
        # get neg antidiagonal
        self.matchLine((-1,1), pos[0]-index, pos[1]+index, -1)
    
    def matchLine(self, lineType:tuple, row, col, direction):
        """
        This function adds the score for a specific line type given a pattern, start point and direction
        args:
            pattern: the string to be compared
            lineType: (rowShift, colShift) | (1,0) = vertical | (0,1) = horizontal | (1,1) = diagonal | (-1, 1) = antiDiagonal
            row: the row pattern[0] is in
            col: the col pattern[0] is in
            direction: 1 = left to right/ top to bottom or top right | -1 = reverse that
        
        returns:
            adds the value of the pattern to the value of the position
            adds the start coords and end coords to the patternMatches history

        ## this maybe could be optimized by considering same direction symmetries at same time but I don't think so since it still has to do all the key comparisons, might only save 1
        """

        ## way to optimize by stopping subsets here

        rowInc = (direction*lineType[0])
        colInc = (direction*lineType[1])
        
        startRow = row
        startCol = col


        row -= rowInc
        col -= colInc

        for point in self.pattern:
            # go to the next row and col
            row += rowInc
            col += colInc

            # check if the point is out of bounds
            if row < 0 or row >= self.height or col < 0 or col >= self.width:
                # if out of bounds and not a wall or * then it fails
                if point != "X":
                    return False
            # star must be in boundsyy
            elif point == "X":
                return False
            elif point == "*":
                continue
            # must be in bounds so no error thrown
            elif point == "P":
                if self.board[row][col] != self.player:
                    return False
            elif point == "O":
                if self.board[row][col] != self.opp:
                    return False
            elif point == "_":
                if self.board[row][col] != 0:
                    return False
            else:
                assert False, f"{point} | got a wrong pattern match"
            
        # the pattern matches
        # check if the pattern is a subset
        for match in self.patternMatches:
            # checks for horizontal and vertical
            # check if line1 is a subset of line2
            if self.is_line_subset(match[0], match[1], (startRow, startCol), (row,col), match[2], lineType):
                # i don't have to care about tie breakers because that has already been handled
                return False

        self.patternMatches.append(((startRow, startCol), (row, col), lineType))
        self.value += self.patVal


    def is_line_subset(self, line2_start, line2_end, subsetStart, subsetEnd, direction1, direction2):
        """
        Check if line1 is a subset of line2.
        Lines can only be: vertical, horizontal, diagonal, or anti-diagonal.
        
        Returns True if line1 is entirely contained within line2.
        """
        
        # First check: Must be the same line type (same direction)
        if direction1 != direction2:
            return False
        
        # Second check: Must be collinear (on the same infinite line)
        if not self.are_collinear(subsetStart, subsetEnd, line2_start, line2_end, direction1):
            return False
        
        # Third check: line1's endpoints must be within line2's range
        return self.is_segment_within(subsetStart, subsetEnd, line2_start, line2_end)


    def are_collinear(self,p1, p2, q1, q2, direction):
        """Check if two line segments are on the same infinite line."""
        if direction == (1,0):
            return p1[1] == q1[1]  # Same col coordinate
        
        elif direction == (0,1):
            return p1[0] == q1[0]  # Same row coordinate
        
        elif direction == (1,1):
            # For diagonal: y - x must be constant
            return (p1[1] - p1[0]) == (q1[1] - q1[0])
        
        elif direction == (-1,1):
            # For antidiagonal: y + x must be constant
            return (p1[1] + p1[0]) == (q1[1] + q1[0])


    def is_segment_within(self, p1, p2, q1, q2):
        """
        Check if segment [p1, p2] is within segment [q1, q2].
        Assumes they're already collinear and same direction.
        """
        # Normalize: ensure start < end for both segments
        # Use the dimension that changes (x for horizontal, y for vertical, etc.)
        
        if p1[1] != p2[1]:  # Non-vertical: use x coordinate
            p_min, p_max = sorted([p1[1], p2[1]])
            q_min, q_max = sorted([q1[1], q2[1]])
            return q_min <= p_min and p_max <= q_max
        
        else:  # Vertical: use y coordinate
            p_min, p_max = sorted([p1[0], p2[0]])
            q_min, q_max = sorted([q1[0], q2[0]])
            return q_min <= p_min and p_max <= q_max


    def findBestType(self, pattern):
        """
        Find the type of move that has the least amount of possible comparisons
        Returns:
            char bestType: "P"or"O" = player | "X" = walls | "_" = empty tiles
        """
        #! assert the num of moves increases when a move is played
        # if num moves = odd then p1 has more
        # if num moves = even then even number of moves
        # if num moves < numTiles - numMoves then empty does not matter
        
        # what there is to compare: empty spaces, walls(perimeter)+4, number of current player, num of opposing player

        hasWalls = False if pattern.find("X") == -1 else True
        hasPlayer = False if pattern.find("P") == -1 else True
        hasOpp = False if pattern.find("O") == -1 else True
        hasEmpty = False if pattern.find("_") == -1 else True

        # does not have any rotations to consider
        # add punishment for to all for the number of cells to check to consider all possible locations
        numWalls = (self.width*6 + self.height*6 + 4) + self.width*2 + self.height*2 + 4
        # multiply by 8 because we would need to consider all 8 rotations
        numEmpty = (self._NUMTILES - self._numMoves) * 8 + self._NUMTILES
        numMoves = ((self._numMoves+1)//2) * 8 + self._numMoves # num moves p1
        numMovesOpp = (self._numMoves//2) * 8 + self._numMoves # num moves p2
        least = min(numWalls, numEmpty, numMoves, numMovesOpp)

        if self.player == 2:
            tmp = numMovesOpp
            numMovesOpp = numMoves
            numMoves = tmp

        if least == numWalls and hasWalls:
            return "X"
        if least == numMoves and hasPlayer:
            return "P"
        if least == numMovesOpp and hasOpp:
            return "O"
        if least == numEmpty and hasEmpty:
            return "_"
    

if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
    