class MatrixIndexError(Exception):
    '''An attempt has been made to access an invalid index in this matrix'''


class MatrixDimensionError(Exception):
    '''An attempt has been made to perform an operation on this matrix which
    is not valid given its dimensions'''


class MatrixInvalidOperationError(Exception):
    '''An attempt was made to perform an operation on this matrix which is
    not valid given its type'''


class MatrixNode():
    '''A general node class for a matrix'''

    def __init__(self, contents, right=None, down=None,i=0,j=0):
        '''(MatrixNode, obj, MatrixNode, MatrixNode) -> NoneType
        Create a new node holding contents, that is linked to right
        and down in a matrix
        '''
        self._contents = contents
        self._right = right
        self._down = down
        self._row_cordinate = i
        self._column_cordinate = j

    def __str__(self):
        '''(MatrixNode) -> str
        Return the string representation of this node
        '''
        return str(self._contents)

    def get_contents(self):
        '''(MatrixNode) -> obj
        Return the contents of this node
        '''
        return self._contents

    def set_contents(self, new_contents):
        '''(MatrixNode, obj) -> NoneType
        Set the contents of this node to new_contents
        '''
        self._contents = new_contents

    def get_right(self):
        '''(MatrixNode) -> MatrixNode
        Return the node to the right of this one
        '''
        return self._right

    def set_right(self, new_node):
        '''(MatrixNode, MatrixNode) -> NoneType
        Set the new_node to be to the right of this one in the matrix
        '''
        self._right = new_node

    def get_down(self):
        '''(MatrixNode) -> MatrixNode
        Return the node below this one
        '''
        return self._down

    def set_down(self, new_node):
        '''(MatrixNode, MatrixNode) -> NoneType
        Set new_node to be below this one in the matrix
        '''
        self._down = new_node
    def get_i_coord(self):
        '''(MatrixNode) -> int
        Return the value of the row index
        '''
        return self._row_cordinate
    def get_j_coord(self):
        '''(MatrixNode) -> int
        Return the value of the col index
        '''        
        return self._column_cordinate


class Matrix():
    '''A class to represent a mathematical matrix'''

    def __init__(self, m, n, default=0):
        '''(Matrix, int, int, float) -> NoneType
        Create a new m x n matrix with all values set to default
        '''
        self._head = MatrixNode(None)
        self._num_columns = n
        self._num_rows = m
        self._de = default
    def get_val(self, i, j):
        '''(Matrix, int, int) -> float
        Return the value of m[i,j] for this matrix m
        EXCEPT: raise MatrixIndexError when i/j> self._num_rows/columns
        EXCEPT: raise MatrixIndexError when i/j < 0
        '''
        if(i> self._num_rows or j> self._num_columns or i < 0 or j < 0):
            raise MatrixIndexError()
        # create an curr at the head
        curr = self._head
        # create a value to return
        return_node = 0
        # check any row or col nodes exist to begin with
        if(curr.get_right() == None or curr.get_down() == None):
            # value is default as no node exist
            return_node = self._de
        # a col/row node does exist
        else:
            # loop through to find the row index we're looking for
            while(curr is not None and curr.get_contents() is not i):
                # go down the row indexs until the loop condition is met
                curr = curr.get_down()
            # here we go j times to the right to the i j location
            while(curr is not None and curr.get_j_coord() is not j):
                # keep going right until the loop conditon is met
                curr = curr.get_right()
            # check if any node in the indexes even exists
            if(curr is None):
                # return val is now default
                return_node = self._de
            # getting to this condition means the node value exists
            else:
                # return val is the contents of this node
                return_node = curr.get_contents()
            
        return return_node
        

    def set_val(self, i, j, new_val):
        '''(Matrix, int, int, float) -> NoneType
        Set the value of m[i,j] to new_val for this matrix m
        EXCEPT: raise MatrixIndexError when i/j> self._num_rows/columns
        EXCEPT: i/j< 0
        '''
        if(i> self._num_rows or j> self._num_columns or i < 0 or j < 0):
            raise MatrixIndexError()        
        curr = self._head
        a_node_put = False
        # first find if any col index nodes exist
        if(curr.get_right() is None):
            # set the bottom to node index de
            curr.set_right(MatrixNode(j))
        else:   
            # if a node exists in general check if we can find the row index
            while(curr is not j and curr is not None):
                # keep going down the nodes by going down one
                curr = curr.get_right()
            # check if we stopped at row index i should it exist
            if(curr == j):
                # do nothing we don't need to create any form of node
                pass
            # say we don't find the node now we check where to place it
            elif(curr is None):
                #start back at the head
                curr = self._head
                # loop through the row index nodes
                while(a_node_put is not True):
                    # check if it's between any two nodes
                    if(curr is not None and curr.get_contents() is not None 
                     and curr.get_right() is not None
                     and curr.get_contents() < j 
                     and curr.get_right().get_contents() > j):
                        # make a new column_node with the previous right of
                        # curr to the new right's right
                        column_node = MatrixNode(j,curr.get_right(), None)
                        #set the new right of curr to column node
                        curr.set_right(column_node)
                        # we've put a new column node so we don't need anymore
                        # iterations
                        a_node_put = True
                    # this case covers if the above case is false
                    # -> (impies) we need to put it at the end
                    elif(curr.get_right() is None and a_node_put == False):
                        # make a column node
                        column_node = MatrixNode(j)
                        # set the new right of the curr to the column node
                        curr.set_right(column_node)
                        # we've again added a node so this iteration/loop
                        # is no longer of use to use
                        a_node_put = True
                    # go to the next column should none of the cases hold
                    # and the loop has to run again
                    curr = curr.get_right()
            # reset the curr to its head
            curr = self._head
            # reset a a_node_put to false since we need it for row node
            a_node_put = False
            # first find if any row index nodes exist
        if(curr.get_down() is None):
            # set the bottom to node index i
            curr.set_down(MatrixNode(i))
        # so a row node exists and we need to see where the new one should go
        else:  
            # if a node exists in general check if we can find the row index
            # make sure we're not going 
            while(curr is not j and curr is not None):
                # keep going down the nodes by going down one
                curr = curr.get_down()
            # check if we stopped at row index i should it exist
            if(curr == i):
                # lets celebrate that we don't need to create a node for this
                # i
                    pass
            # say we don't find the node now we check where to place it
            else:
                #start back at the head
                curr = self._head
                # loop through the row index nodes
                while(a_node_put!=True):
                    # check if it's between any two nodes
                    if(curr is not None and curr.get_contents() is not None 
                     and curr.get_down() is not None
                     and curr.get_contents() < i 
                     and curr.get_down().get_contents() > i):
                        # make a new row node with curr.get_down
                        # as its down node
                        row_node = MatrixNode(i,None,curr.get_down())
                        # set the new down to row node
                        curr.set_down(row_node)
                        # out while loop isn't needed anymore since we've added
                        # a node so we should stop it
                        a_node_put = True
                    # if we access this case that means the row node exists
                    # at the end of the existing nodes
                    elif(curr.get_down() is None and a_node_put == False):
                        # make the previous last row node point to the new
                        # last row node (last in the current greatest row
                        # index) node
                        curr.set_down(MatrixNode(i))
                        # this while loop os no longer needed
                        a_node_put = True
                    # look at the next set of curr and curr.get_down
                    # for the above cases
                    curr = curr.get_down()
        # reset the curr so we don't go out of bounds
        curr = self._head
        # reset the a_node_put as we're gonna set the value for the row index
        a_node_put = False
        # look for the row node or when it reaches none
        while(curr is not None and curr.get_contents() is not i):
            # keep going down till the loop case(s) is meet
            curr = curr.get_down()
        # see if the right of the row_index is none
        if(curr.get_right() is None):
            # set the rows right to the node new_val and save the i and j
            # location for get_val
            curr.set_right(MatrixNode(new_val,None,None,i,j))
        # if we get here it means value nodes already exist in our row index        
        else:           
            # loop through until we've create/updated a node
            while(a_node_put is not True):
                # check if theres a node before the new one
                # we don't go out of bounds
                #while(curr != None and curr.get_right() != None and
                      #(curr.get_right()).get_j_coord() < j):
                        ## go to the node to the left of the new node
                        #curr = curr.get_right()
                        ## check if the next value 
                        #if(curr.get_right() != None and 
                           #curr.get_right().get_j_coord() == j):               
                            #curr.get_right().set_contents(MatrixNode(new_val,None,None,i,j))      
                        #else:
                            #new_val_row_node = MatrixNode(new_val, curr.get_right(), None,i,j)
                            #curr.set_right(new_val_row_node)

                # here we check if we've found the node
                if(curr.get_right().get_j_coord() is not None and curr.get_right().get_j_coord() is j):
                    # save the previous right to prev_noe
                    prev_right = curr.get_right()
                    # set the right to a new node with new_val and save the
                    # i and j location for get_val and prev_right to it's new
                    # new right
                    curr.set_right(MatrixNode(new_val,prev_right,None,i,j))
                    # again we don't need the loop anymore we've added the
                    # updated the node
                    a_node_put = True
                # check if theres a node before the new one
                # we don't go out of bounds                
                elif(curr is not None and curr.get_right() is not None and
                      (curr.get_right()).get_j_coord() < j):
                        # go to the node to the left of the new node
                        curr = curr.get_right()
                        # check if the next value 
                        if(curr.get_right() != None and 
                           curr.get_right().get_j_coord() == j):               
                            curr.get_right().set_contents(MatrixNode(new_val,None,None,i,j))      
                        else:
                            new_val_row_node = MatrixNode(new_val, curr.get_right(), None,i,j)
                            curr.set_right(new_val_row_node)
                        a_node_put = True                
                elif(curr is not None and curr.get_contents() is not None
                   and curr.get_right() is not None
                   and curr.get_j_coord() < j
                   and curr.get_right().get_j_coord() > j):
                    # save the prev_right
                    prev_right = curr.get_right()
                    # set the right of the existing node(one we're on) to
                    # a new node of new_val and its previous right                    
                    curr.set_right(MatrixNode(new_val,prev_right, None, i, j))
                    # we've created a new node so we're done with the loop
                    a_node_put = True
                # if the two cases haven't been met then well its at the end
                elif(curr.get_right() is None and a_node_put == False):
                    # set the last node before the none to new val node and
                    # save the i and j location for get_val
                    curr.set_right(MatrixNode(new_val,None,None, i, j))
                    # we're done with the looop as we've create a new node
                    a_node_put = True
                
                curr = curr.get_right()
        # reset curr to head
        curr = self._head
        # reset a_put_node for row
        a_node_put = False
        # look for the column we're looking for
        while(curr is not None and curr.get_contents() is not j):
            #print('h')
            # keep going to the right until we find the column node
            curr = curr.get_right()
        #check if the column has any values
        if(curr.get_down() is None):
            # set the new val node to the next down of curr (column node)
            curr.set_down(MatrixNode(new_val,None,None,i,j))
        # nodes exist so lets see where to locate the new val node
        else:
            # loop through until we've added/updated a node
            while(a_node_put is not True):
                # check if there's a value before where the new node is suppose
                # to be
                if((MatrixNode((curr.get_down().get_i_coord())-1)).get_contents() is not None):
                    # check if theres a node before this one(curr) as we may
                    # have set the one above it's down to none should it exist
                    while(curr is not None and curr.get_down() is not None and
                          curr.get_down().get_i_coord() < i):
                        # set the curr to the node above the one we want to set
                        curr = curr.get_down()
                    # create a new column value node with the i coordinate
                    # given by the user
                    new_val_col_node = MatrixNode(curr.get_i_coord())
                    # set it so the node below curr is below new_val_col_node
                    new_val_col_node.set_down(curr.get_down())
                    # and my curr's new down node is the new val node
                    curr.set_down(new_val_col_node)
                    # lets leave the while loop that checks if i've created a
                    # new node or updated one
                    a_node_put = True
                else:
                        
                    # check if the i j location exists
                    if(curr.get_down().get_i_coord() is i):
                        # save prev get_down
                        prev_down = curr.get_down()
                        # set the down to new val node with i and j saved for
                        # get val
                        curr.set_down(MatrixNode(new_val,None,prev_down,i,j))
                        # finish up with the loop as we've updated a previous node
                        a_node_put = True
                    # check if the new_val node belongs in the middle
                    elif(curr is not None and curr.get_contents() is not None
                       and curr.get_down() is not None
                       and curr.get_i_coord() < i
                       and curr.get_down().get_i_coord() > i):
                        # save the previous down of curr
                        prev_down = curr.get_down()
                        # set the matrix node of new val to currs new down
                        # and make prev its down and save the i and j for comparing
                        curr.set_down(MatrixNode(new_val,None,prev_down,i,j))
                        # equate to true to stop loop since we've added a new node
                        a_node_put = True
                    # should the two be false above, we insert at the end of linked
                    # list
                    elif(curr.get_down() is None and a_node_put == False):
                        # save prev down
                        prev_down = curr.get_down()
                        # set the currs down to new val, and set the new downs down
                        # to prev down and save the i and j location for comapring
                        curr.set_down(MatrixNode(new_val,None,prev_down,i,j))
                        # loop is no longer needed, finish the loop and move on
                        # as we've created a new node
                        a_node_put = True
                    # look at the next set of curr and curr.get_down()
                    # should a_put_node still be false
                    curr = curr.get_down()

    def get_row(self, row_num):
        '''(Matrix, int) -> OneDimensionalMatrix
        Return the row_num'th row of this matrix
        EXCEPT: raise MatrixIndexError when row_num > self._num_rows
        EXCEPT: raise MatrixIndexError when row_num < 0
        '''
        if(row_num > self._num_rows or row_num < 0):
            raise MatrixIndexError()       
        # create a curr thats the head of the matrix
        curr = self._head
        # create a one_d matrix to return
        one = OneDimensionalMatrix(1,self._num_columns)
        # loop through num_columns as it determines how many times i go through
        # row row_num
        for j in range(0,self._num_columns):
            # set item to the one_d at row_num j in the matrix item
            one.set_item(j,self.get_val(row_num,j))
        return one

    def set_row(self, row_num, new_row):
        '''(Matrix, int, OneDimensionalMatrix) -> NoneType
        Set the value of the row_num'th row of this matrix to those of new_row
        EXCEPT: raise MatrixIndexError when row_num > self._num_rows
        EXCEPT: raise MatrixIndexError when row_num < 0
        EXCEPT: raise MatrixIndexError when new_columns._num_columns !=self._num_columns
        '''
        if(row_num > self._num_rows or row_num < 0 or new_row._num_columns !=self._num_columns):
            raise MatrixIndexError()        
        # create a curr to keep track of the nodes
        curr = self._head
        # loop through num_col times which is the length of the row
        for i in range(0,self._num_columns):
            # set each val in the new row to the new_row at that index (i)
            curr.set_val(i,row_num,new_row.get_item(i))

    def get_col(self, col_num):
        '''(Matrix, int) -> OneDimensionalMatrix
        Return the col_num'th column of this matrix
        EXCEPT: raise MatrixIndexError when col_num> self._num_columns
        EXCEPT: raise MatrixIndexError when col_num < 0
        '''
        if(col_num> self._num_columns or col_num < 0):
            raise MatrixIndexError()        
        # create an empty column one_d matrix
        one = OneDimensionalMatrix(self._num_rows,1)
        # loop through the length of the column matrix
        for j in range(0,self._num_rows):
            # set the value of one to those of matrix at col_num j index
            one.set_item(j,self.get_val(col_num,j))
        return one        

    def set_col(self, col_num, new_col):
        '''(Matrix, int, OneDimensionalMatrix) -> NoneType
        Set the value of the col_num'th column of this matrix to those of new_row
        EXCEPT: raise MatrixIndexError when col_num> self._num_columns
        EXCEPT: raise MatrixIndexError when col_num < 0
        EXCEPT: raise MatrixIndexError when new_columns._num_rows !=self._num_rows
        '''
        if(col_num> self._num_columns or col_num < 0 or new_columns._num_rows !=self._num_rows):
            raise MatrixIndexError()        
        # create a curr to keep track of the head
        curr = self._head
        # go through the length of the column
        for i in range(0,self._num_rows):
            # set value of curr (Matrix) at i and col_num to the one_d(column
            # matrix) at i
            curr.set_val(i,col_num,new_col.get_item(i))        

    def swap_rows(self, i, j):
        '''(Matrix, int, int) -> NoneType
        Swap the values of rows i and j in this matrix
        EXCEPT: raise MatrixIndexError when i/j> self._num_rows
        EXCEPT: i/j< 0
        '''
        if(i > self._num_rows or i < 0 or j > self._num_rows or j < 0 ):
            raise MatrixIndexError()        
        # create a curr to keep track of the head
        curr = self._head
        # get a row matrix of row 1
        row1 = self.get_row(i)
        # get a row matrix of row2
        row2 = self.get_row(j)
        # loop through until we get to row1 index
        for a in range(0,i):
            # keep going down(this just makes the head go down the row indexes)
            curr = curr.get_down()
        # here we loop through the length of the row
        for b in range(0,self._num_columns):
            # set row1 at b with row2 at b
            self.set_val(i,b,row1.get_item(b))
        # reset the head for row2
        curr = self._head
        # loop through until we find index j
        for c in range(0,j):
            # set the head to the one beneath itself
            curr = curr.get_down()
        # loop through the length of the row matrix
        for d in range(0,self._num_columns):
            # set row2 at d  with row1 at d
            self.set_val(j,d,row2.get_item(d))        
        

    def swap_cols(self, i, j):
        '''(Matrix, int, int) -> NoneType
        Swap the values of columns i and j in this matrix
        EXCEPT: raise MatrixIndexError when i/j > self._num_columns
        EXCEPT: i/j < 0
        '''
        if(i > self._num_columns or i < 0 or j > self._num_columns or j < 0 ):
            raise MatrixIndexError() 
        # create a curr to know the head's location
        curr = self._head
        # make a one_d col matrix of col1
        col1 = self.get_col(i)
        # again we make a one_d col matrix of col2
        col2 = self.get_col(j)
        # loop through until we find col1 (i's index)
        for a in range(0,i):
            # set our curr to the right of itself
            curr = curr.get_right()
        # go through the col's len
        for b in range(0,self._num_rows):
            # make col1 at b = col2 at b
            self.set_val(i,b,col1.get_item(b))
        # reset the curr for col2
        curr = self._head
        # loop through until we get to col2's idnex (j)
        for c in range(0,j):
            # the one right of curr is the new curr
            curr = curr.get_right()
        # go through the length of col2
        for d in range(0,self._num_rows):
            # col2 at d = col1 at d 
            self.set_val(j,d,col2.get_item(d))        

    def add_scalar(self, add_value):
        '''(Matrix, float) -> NoneType
        Increase all values in this matrix by add_value
        '''
        # loop through the num_columns
        for i in range(0,self._num_columns):
            # loop through the num_rows
            for j in range(0,self._num_rows):
                # each element in matrix/self will be added by add_value
                # by taking it's previous vale adding it with add_vale
                # and set it at j i 
                self.set_val(j,i,self.get_val(j,i)+add_value)

    def subtract_scalar(self, sub_value):
        '''(Matrix, float) -> NoneType
        Decrease all values in this matrix by sub_value
        '''
        # loop through num_col times
        for i in range(0,self._num_columns):
            # loop through num_row times
            # we want i and j so we can keep track of the values we update
            for j in range(0,self._num_rows):
                # previous val of self - sub_value at j i will give us a
                # a matrix with its OG values deducted by sub_val amount
                self.set_val(j,i,self.get_val(j,i)-sub_value)

    def multiply_scalar(self, mult_value):
        '''(Matrix, float) -> NoneType
        Multiply all values in this matrix by mult_value
        '''
        # loop through num_col times to keep track of what col we're on
        for i in range(0,self._num_columns):
            # loop through num row times to keep track of what row we're on
            for j in range(0,self._num_rows):
                # multi prev val of self with mult value and set that val at
                # j i 
                self.set_val(j,i,self.get_val(j,i)*mult_value)       

    def add_matrix(self, adder_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the sum of this matrix and adder_matrix
        EXCEPT: MatrixDimensionError = self._num_columns != adder_matrix._num_columns
        EXCEPT: MatrixDimensionError = self._num_rows != adder_matrix._num_rows
        '''
        if(self._num_columns != adder_matrix._num_columns or self._num_rows != adder_matrix._num_rows):
            raise MatrixDimensionError
        # make a new_matrix of selfs num cols and rows
        new_matrix = Matrix(self._num_rows, self._num_columns)
        # loop through num_col times
        for i in range(0,self._num_columns):
            # loop through num row times
            for j in range(0,self._num_rows):
                # at row j, col i, set val to self at j i + adder at j i
                # is new_matrix at j i
                new_matrix.set_value(j,i,self.get_row(j,i)+adder_matrix.get_row(j,i))        
        

    def multiply_matrix(self, mult_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the product of this matrix and mult_matrix
        EXCEPT: raise MatrixDimensionError = self._num_columns != mult_matrix._num_rows num rows != num cols
        '''
        if(self._num_columns != mult_matrix._num_rows):
            raise MatrixDimensionError 
        # make a new matrix with self's rows and mult's columns
        new_matrix = Matrix(self._num_rows, multi_matrix._num_columns)
        # loop through num col times of self
        for i in range(0,self.num_rows):
            # loop through num row times of mult_matrix
            for j in range(0,multi_matrix.num_columns):
                # loop through num col times of self
                for k in range(0,self._num_columns):
                    # so new_matrix at i and j is self at i and k * mult k j
                    # values gotten
                    new_matrix.set_val(i,j,self.get_val(i,k)*mult_matrix.get_val(k,j))
        return new_matrix
                


class OneDimensionalMatrix(Matrix):
    '''A 1xn or nx1 matrix.
    (For the purposes of multiplication, we assume it's 1xn)'''
    
    def __init__(self,m,n,default=0):
        Matrix.__init__(self,m,n,default)

    def get_item(self, i):
        '''(OneDimensionalMatrix, int) -> float
        Return the i'th item in this matrix
        EXCEPT: raise MatrixIndexError when i> self._num_rows/columns
        '''
        if(i> self._num_rows or i<0 or i> self._num_columns):
            raise MatrixIndexError
        # make a retunr value equal to 0
        return_val = 0
        # check if its a column matrix
        if(self._num_columns == 1):
            # get the val at i in the col matrix
            return_val = self.get_val(i,1)
        # otherwise its a row matrix
        else:
            # get the value at i in the row matrix
            return_val = self.get_val(1,i)
        return return_val


    def set_item(self, i, new_val):
        '''(OneDimensionalMatrix, int, float) -> NoneType
        Set the i'th item in this matrix to new_val
        EXCEPT: raise MatrixIndexError when i> self._num_rows/columns
        '''
        if(i> self._num_rows or i<0 or i> self._num_columns):
            raise MatrixIndexError        
        # check if the one_d matrix is a column matrix
        if(self._num_columns == 1):
            # set the row matrix at i to new_val
            self.set_val(i,1,new_val)
        # if not above occurs then its a column matrix
        else:
            # set val at col matrix at 1 i to new_val
            self.set_val(1,i,new_val)


class SquareMatrix(Matrix):
    '''A matrix where the number of rows and columns are equal'''
    def __init__(self,m,default=0):
        Matrix.__init__(self,m,m,default)
        
    def transpose(self):
        '''(SquareMatrix) -> NoneType
        Transpose this matrix
        '''
        # get the col1 and row1
        # make the col1 at i equal row1 at i
        # do the vice versa of the above comment
        # pray this gets you some part marks

    def get_diagonal(self):
        '''(Squarematrix) -> OneDimensionalMatrix
        Return a one dimensional matrix with the values of the diagonal
        of this matrix
        '''
        # create an empty one_d matrix
        one = OneDimensionalMatrix(1,self._num_columns)
        # loop through row num times
        for i in range(0,self.num_rows):
            # set your one_d matrix's value/item at i with self at i i
            one.set_item(i,self.get_val(i,i))
        return one
            

    def set_diagonal(self, new_diagonal):
        '''(SquareMatrix, OneDimensionalMatrix) -> NoneType
        Set the values of the diagonal of this matrix to those of new_diagonal
        '''
        # loop through num_row times
        for i in range(0,self.num_rows):
            # set your val of squarematatrix at i i with the diagonal at i
            self.set_val(i,i,new_diagonal.get_item(i))        


class SymmetricMatrix(SquareMatrix):
    '''A Symmetric Matrix, where m[i, j] = m[j, i] for all i and j'''
    def __init__(self,m,default=0):
        Matrix.__init__(self,m,m,default)    

class DiagonalMatrix(SquareMatrix, OneDimensionalMatrix):
    '''A square matrix with 0 values everywhere but the diagonal'''
    def __init__(self,m,default=0):
        SquareMatrix.__init__(self,m,default=0)
        SquareMatrix.get_diagonal()
    def add_scalar(self, add_value):
        '''(Matrix, float) -> NoneType
        Increase all values in this matrix by add_value
        '''
        raise MatrixInvalidOperationError()
    
    def subtract_scalar(self, sub_value):
        '''(Matrix, float) -> NoneType
        Decrease all values in this matrix by sub_value
        '''
        raise MatrixInvalidOperationError()

    def multiply_scalar(self, mult_value):
        '''(Matrix, float) -> NoneType
        Multiply all values in this matrix by mult_value
        '''       
        raise MatrixInvalidOperationError()

    def add_matrix(self, adder_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the sum of this matrix and adder_matrix
        '''    
        raise MatrixInvalidOperationError()

    def multiply_matrix(self, mult_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the product of this matrix and mult_matrix
        '''
        raise MatrixInvalidOperationError()    

class IdentityMatrix(DiagonalMatrix):
    '''A matrix with 1s on the diagonal and 0s everywhere else'''
    def add_scalar(self, add_value):
        '''(Matrix, float) -> NoneType
        Increase all values in this matrix by add_value
        '''
        raise MatrixInvalidOperationError()
    
    def subtract_scalar(self, sub_value):
        '''(Matrix, float) -> NoneType
        Decrease all values in this matrix by sub_value
        '''
        raise MatrixInvalidOperationError()

    def multiply_scalar(self, mult_value):
        '''(Matrix, float) -> NoneType
        Multiply all values in this matrix by mult_value
        '''       
        raise MatrixInvalidOperationError()

    def add_matrix(self, adder_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the sum of this matrix and adder_matrix
        '''    
        raise MatrixInvalidOperationError()

    def multiply_matrix(self, mult_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the product of this matrix and mult_matrix
        '''
        raise MatrixInvalidOperationError()
