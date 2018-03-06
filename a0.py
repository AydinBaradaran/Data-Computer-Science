PUZZLE1 = '''
glkutqyu
onnkjoaq
uaacdcne
gidiaayu
urznnpaf
ebnnairb
xkybnick
ujvaynak
'''

PUZZLE2 = '''
fgbkizpyjohwsunxqafy
hvanyacknssdlmziwjom
xcvfhsrriasdvexlgrng
lcimqnyichwkmizfujqm
ctsersavkaynxvumoaoe
ciuridromuzojjefsnzw
bmjtuuwgxsdfrrdaiaan
fwrtqtuzoxykwekbtdyb
wmyzglfolqmvafehktdz
shyotiutuvpictelmyvb
vrhvysciipnqbznvxyvy
zsmolxwxnvankucofmph
txqwkcinaedahkyilpct
zlqikfoiijmibhsceohd
enkpqldarperngfavqxd
jqbbcgtnbgqbirifkcin
kfqroocutrhucajtasam
ploibcvsropzkoduuznx
kkkalaubpyikbinxtsyb
vjenqpjwccaupjqhdoaw
'''


def rotate_puzzle(puzzle):
    '''(str) -> str
    Return the puzzle rotated 90 degrees to the left.
    '''

    raw_rows = puzzle.split('\n')
    rows = []
    # if blank lines or trailing spaces are present, remove them
    for row in raw_rows:
        row = row.strip()
        if row:
            rows.append(row)

    # calculate number of rows and columns in original puzzle
    num_rows = len(rows)
    num_cols = len(rows[0])

    # an empty row in the rotated puzzle
    empty_row = [''] * num_rows

    # create blank puzzle to store the rotation
    rotated = []
    for row in range(num_cols):
        rotated.append(empty_row[:])
    for x in range(num_rows):
        for y in range(num_cols):
            rotated[y][x] = rows[x][num_cols - y - 1]

    # construct new rows from the lists of rotated
    new_rows = []
    for rotated_row in rotated:
        new_rows.append(''.join(rotated_row))

    rotated_puzzle = '\n'.join(new_rows)

    return rotated_puzzle


def lr_occurrences(puzzle, word):
    '''(str, str) -> int
    Return the number of times word is found in puzzle in the
    left-to-right direction only.

    >>> lr_occurrences('xaxy\nyaaa', 'xy')
    1
    '''
    return puzzle.count(word)

# ---------- Your code to be added below ----------

# *task* 3: write the code for the following function.
# We have given you the header, type contract, example, and description.


def total_occurrences(puzzle, word):
    '''(str, str) -> int
    Return total occurrences of word in puzzle.
    All four directions are counted as occurrences:
    left-to-right, top-to-bottom, right-to-left, and bottom-to-top.

    >>> total_occurrences('xaxy\nyaaa', 'xy')
    2
    '''
    # your code here
    # check if the word occurs left-to-right save to L_to_R
    L_to_R = lr_occurrences(puzzle, word)
    # check if the word occurs right-to-left save to R_to_L
    R_to_L = lr_occurrences(rotate_puzzle(rotate_puzzle(puzzle)), word)
    # check if the word occurs top-to-bottom save to top_to_bottom
    top_to_bottom = lr_occurrences(rotate_puzzle(puzzle), word)
    # check if the word occurs bottom-to-top save to bottom_to_top
    bottom_to_top = lr_occurrences(rotate_puzzle(rotate_puzzle
                                                 (rotate_puzzle(puzzle))),
                                   word)
    # add up all the directions of word and return the amount of times
    # it occurs
    return (L_to_R+R_to_L+top_to_bottom+bottom_to_top)
# *task* 5: write the code for the following function.
# We have given you the function name only.
# You must follow the design recipe and complete all parts of it.
# Check the handout for what the function should do.


def in_puzzle_horizontal(puzzle, word):
    '''(str,str) -> bool
    user will input the string/puzzle with the designated word\
    he/she is lookng for left-to-right, and right-to-left and if\
    it can be found more than once including 1, true will be returned\
    meaning we found it from either side (or both) as previously mentioned,\
    and false if we cant find it at all (word DNE as a possibility) from\
    either side (or both).
    REQ: string is inputed for both the puzzle variable and word variable
    REQ: string inputed matches the exact word in the puzzle (case sensitive)
    >>>in_puzzle_horizontal(PUZZLE1, "brian")
    True
    >>>in_puzzle_horizontal(PUZZLE1, "nick")
    True
    >>>in_puzzle_horizontal(PUZZLE1, "dan")
    False
    >>>in_puzzle_horizontal(PUZZLE1, "BRIAN")
    False
    >>>in_puzzle_horizontal(PUZZLE1, "an")
    True
    >>>in_puzzle_horizontal(PUZZLE1,"anya")
    True
    >>>in_puzzle_horizontal(PUZZLE1,"paco")
    False
    >>>in_puzzle_horizontal(PUZZLE2, "brian")
    False
    >>>in_puzzle_horizontal(PUZZLE2, "nick")
    True
    >>>in_puzzle_horizontal(PUZZLE2, "dan")
    False
    >>>in_puzzle_horizontal(PUZZLE2, "BRIAN")
    False
    >>>in_puzzle_horizontal(PUZZLE2, "an")
    True
    >>>in_puzzle_horizontal(PUZZLE2,"anya")
    True
    >>>in_puzzle_horizontal(PUZZLE2,"paco")
    False
    '''
    # check if the word occurs left to right save to L_to_R
    L_to_R = lr_occurrences(puzzle, word)
    # check if the word occurs right to left and save to R_to_L
    R_to_L = lr_occurrences(rotate_puzzle(rotate_puzzle(puzzle)), word)
    # add the two above if its >=1 return True if not then False
    return ((L_to_R + R_to_L) >= 1)
# *task* 8: write the code for the following function.
# We have given you the function name only.
# You must follow the design recipe and complete all parts of it.
# Check the handout for what the function should do.


def in_puzzle_vertical(puzzle, word):
    '''(str,str) -> bool
    user will input the string/puzzle with the designated word\
    he/she is lookng for top-to-bottom, and bottom-to-top and\
    if it can be found more than once including 1 (vertically),
    true will be returned meaning we found it from either side\
    (or both) as previously mentioned, and false if we cant find\
    it at all (even if word DNE in general) from either side (or both).
    REQ: string is inputed for both the puzzle variable and word variable
    REQ: string inputed matches the exact word in the puzzle (case sensitive)
    >>>in_puzzle_vertical(PUZZLE1, "brian")
    True
    >>>in_puzzle_vertical(PUZZLE2, "brian")
    True
    >>>in_puzzle_vertical(PUZZLE1, "nick")
    True
    >>>in_puzzle_vertical(PUZZLE2, "nick")
    True
    >>>in_puzzle_vertical(PUZZLE1, "dan")
    True
    >>>in_puzzle_vertical(PUZZLE2, "dan")
    True
    >>>in_puzzle_vertical(PUZZLE1, "BRIAN")
    False
    >>>in_puzzle_vertical(PUZZLE2, "BRIAN")
    False
    >>>in_puzzle_vertical(PUZZLE1, "an")
    True
    >>>in_puzzle_vertical(PUZZLE2, "an")
    True
    >>>in_puzzle_vertical(PUZZLE1,"anya")
    True
    >>>in_puzzle_vertical(PUZZLE2, "anya")
    True
    >>>in_puzzle_vertical(PUZZLE1,"paco")
    True
    >>>in_puzzle_vertical(PUZZLE2, "paco")
    True
    '''
    # check if the word occurs top to bottom and save it to top_to_bottom
    top_to_bottom = lr_occurrences(rotate_puzzle(puzzle), word)
    # check if the word occurs bottom to top and save it save to bottom_to_top
    bottom_to_top = lr_occurrences(rotate_puzzle
                                   (rotate_puzzle(rotate_puzzle(puzzle))),
                                   word)
    # add the two to see if it equals 1 or more then return true or false
    return(top_to_bottom + bottom_to_top >= 1)
# *task* 9: write the code for the following function.
# We have given you the function name only.
# You must follow the design recipe and complete all parts of it.
# Check the handout for what the function should do.


def in_puzzle(puzzle, word):
    '''(str,str) -> bool
    user will input the string/puzzle with the designated word\
    he/she is lookng for top-to-bottom,bottom-to-top,right-to-left\
    and left-to-right if it can be found more than once including 1\
    true will be returned meaning we found it from any side\
    (1 or 2 or 3 or 4 of the sides) as previously mentioned, and\
    false if we cant find it at all from either of the four sides.
    REQ: string is inputed for both the puzzle variable and word variable
    REQ: string inputed matches the exact word in the puzzle (case sensitive)
    >>>in_puzzle(PUZZLE1, "brian")
    True
    >>>in_puzzle(PUZZLE1, "nick")
    True
    >>>in_puzzle(PUZZLE1, "dan")
    True
    >>>in_puzzle(PUZZLE1, "BRIAN")
    False
    >>>in_puzzle(PUZZLE1, "an")
    True
    >>>in_puzzle(PUZZLE1,"anya")
    True
    >>>in_puzzle(PUZZLE1,"paco")
    True
    >>>in_puzzle(PUZZLE2, "brian")
    True
    >>>in_puzzle(PUZZLE2, "nick")
    True
    >>>in_puzzle(PUZZLE2, "dan")
    True
    >>>in_puzzle(PUZZLE2, "BRIAN")
    False
    >>>in_puzzle(PUZZLE2, "an")
    True
    >>>in_puzzle(PUZZLE2,"anya")
    True
    >>>in_puzzle(PUZZLE2,"paco")
    True
    '''
    # call total_occurences that takes all 4 directions into account
    # >=1 True is returned if not then False is returned
    return (total_occurrences(puzzle, word) >= 1)

# *task* 10: write the code for the following function.
# We have given you only the function name and parameters.
# You must follow the design recipe and complete all parts of it.
# Check the handout for what the function should do.


def in_exactly_one_dimension(puzzle, word):
    '''(str,str) -> bool
    user will input the string/puzzle he or she is looking at\
    with the word he or she is looking for and if it appears only\
    in one direction (only horizontal/vertical) then return true\
    signifying it can only be found in one dimension but if its\
    both or it cant be found false is returned signifying it\
    can't be found or it occurs in more than one direction
    REQ: str==str (word by word, case sensitive)
    REQ: a str is inputed for both puzzle and word
    >>>in_exactly_one_dimension(PUZZLE1, "brian")
    False
    >>>in_exactly_one_dimension(PUZZLE1, "nick")
    False
    >>>in_exactly_one_dimension(PUZZLE1, "dan")
    True
    >>>in_exactly_one_dimension(PUZZLE1, "BRIAN")
    False
    >>>in_exactly_one_dimension(PUZZLE1, "an")
    False
    >>>in_exactly_one_dimension(PUZZLE1,"anya")
    False
    >>>in_exactly_one_dimension(PUZZLE1,"paco")
    True
    >>>in_exactly_one_dimension(PUZZLE2, "brian")
    True
    >>>in_exactly_one_dimension(PUZZLE2, "nick")
    False
    >>>in_exactly_one_dimension(PUZZLE2, "dan")
    True
    >>>in_exactly_one_dimension(PUZZLE2, "BRIAN")
    False
    >>>in_exactly_one_dimension(PUZZLE2, "an")
    False
    >>>in_exactly_one_dimension(PUZZLE2,"anya")
    False
    >>>in_exactly_one_dimension(PUZZLE2,"paco")
    True
    '''
    # if one side of in_puzzle_vertical/horizontal is True
    # and the other False and vice versa True is returned
    # if not then False is to be returned
    return (((in_puzzle_vertical(puzzle, word) == True and
              in_puzzle_horizontal(puzzle, word) == False) or
             (in_puzzle_vertical(puzzle, word) == False and
              in_puzzle_horizontal(puzzle, word) == True)))
# *task* 11: write the code for the following function.
# We have given you only the function name and parameters.
# You must follow the design recipe and complete all parts of it.
# Check the handout for what the function should do.


def all_horizontal(puzzle, word):
    '''(str,str) -> bool
    The user will input the designated puzzle/string he/she is looking at\
    and the word he/she is trying to verify within it, if the words occurence\
    is only horizontal or the word doesnt exist, true is returned, if neither\
    of the previously mentioned conditions are met False is returned meaning\
    it occurs vertically and horizontally or it occurs x amount of times\
    vertically
    REQ: str==str (case sensitive)
    REQ: str is inputed
    >>>all_horizontal(PUZZLE1, "brian")
    False
    >>>all_horizontal(PUZZLE1, "nick")
    False
    >>>all_horizontal(PUZZLE1, "dan")
    False
    >>>all_horizontal(PUZZLE1, "BRIAN")
    True
    >>>all_horizontal(PUZZLE1, "an")
    False
    >>>all_horizontal(PUZZLE1,"anya")
    False
    >>>all_horizontal(PUZZLE1,"paco")
    False
    >>>all_horizontal("aabc\naabc","aa")
    False
    >>>all_horizontal(PUZZLE2, "brian")
    False
    >>>all_horizontal(PUZZLE2, "nick")
    False
    >>>all_horizontal(PUZZLE2, "dan")
    False
    >>>all_horizontal(PUZZLE2, "BRIAN")
    True
    >>>all_horizontal(PUZZLE2, "an")
    False
    >>>all_horizontal(PUZZLE2,"anya")
    False
    >>>all_horizontal(PUZZLE2,"paco")
    False
    >>>all_horizontal("abc\nabc","abc")
    True
    '''
    # if in_puzzle_horizontal is True and in_puzzle_vertical
    # is False or the word doesn exist then True is returned
    # if the conditions are met then False is to be returned
    return ((in_puzzle_vertical(puzzle, word) == False and
             in_puzzle_horizontal(puzzle, word) == True) or
            total_occurrences(puzzle, word) == 0)

# *task* 12: write the code for the following function.
# We have given you only the function name and parameters.
# You must follow the design recipe and complete all parts of it.
# Check the handout for what the function should do.


def at_most_one_vertical(puzzle, word):
    '''(str,str) -> bool
    The user will call the following function and check\
    to see if the inputed word occurs at most once if\
    the word doesnt occur True is returned, if the word\
    is found once and the only case it can be found is\
    vertically then True is returned, if the word occurs\
    more than once or if does occurs once but it does so\
    horizontally False is to be returned.
    REQ: str is inputed for both parameters in the function
    REQ: str== str (word exactly matches the word found in\
    the string/puzzle, so case sensitve)
    >>>at_most_one_vertical("abac\nabfc","aa")
    False
    >>>at_most_one_vertical(PUZZLE1, "paco")
    True
    >>>at_most_one_vertical(PUZZLE2, "paco")
    True
    >>>at_most_one_vertical(PUZZLE1, "brian")
    False
    >>>at_most_one_vertical(PUZZLE1, "nick")
    False
    >>>at_most_one_vertical(PUZZLE2, "nick")
    False
    >>>at_most_one_vertical(PUZZLE2, "brian")
    False
    >>>at_most_one_vertical(PUZZLE1, "anya")
    False
    >>>at_most_one_vertical(PUZZLE2, "dan")
    False
    >>>at_most_one_vertical(PUZZLE1, "dan")
    False
    >>>at_most_one_vertical(PUZZLE2, "anya")
    False
    >>>at_most_one_vertical(PUZZLE2, "BRIAN")
    True
    >>>at_most_one_vertical(PUZZLE1, "BRIAN")
    True
    >>>at_most_one_vertical("bbb\nalc", "ba")
    True
    '''
    # if the word occurs once and that occurence is
    # vertical or the total occurnce is 0 then True
    # is returned, if the conditions aren't met then
    # False is to be returned
    return ((total_occurrences(puzzle, word) == 1 and
            in_puzzle_vertical(puzzle, word) == True)or
            total_occurrences(puzzle, word) == 0)


def do_tasks(puzzle, name):
    '''(str, str) -> NoneType
    puzzle is a word search puzzle and name is a word.
    Carry out the tasks specified here and in the handout.
    '''

    # *task* 1a: add a print call below the existing one to print
    # the number of times that name occurs in the puzzle left-to-right.
    # Hint: one of the two starter functions defined above will be useful.

    # the end='' just means "Don't start a newline, the next thing
    # that's printed should be on the same line as this text
    print('Number of times', name, 'occurs left-to-right: ', end='')
    # your print call here
    print(lr_occurrences(puzzle, name))
    # *task* 1b: add code that prints the number of times
    # that name occurs in the puzzle top-to-bottom.
    # (your format for all printing should be similar to
    # the print statements above)
    print('Number of times', name, 'occurs top-to-bottom: ', end='')
    # Hint: both starter functions are going to be useful this time!
    print(lr_occurrences(rotate_puzzle(puzzle), name))
    # *task* 1c: add code that prints the number of times
    print('Number of times', name, 'occurs right-to-left: ', end='')
    # that name occurs in the puzzle right-to-left.
    print(lr_occurrences(rotate_puzzle(rotate_puzzle(puzzle)), name))
    # *task* 1d: add code that prints the number of times
    print('Number of times', name, 'occurs bottom-to-top: ', end='')
    # that name occurs in the puzzle bottom-to-top.
    print(lr_occurrences(rotate_puzzle(rotate_puzzle(rotate_puzzle
          (puzzle))), name))
    # *task* 4: print the results of calling total_occurrences on
    # puzzle and name.
    # Add only one line below.
    # Your code should print a single number, nothing else.
    print (total_occurrences(puzzle, name))
    # *task* 6: print the results of calling in_puzzle_horizontal on
    # puzzle and name.
    # Add only one line below. The code should print only True or False.
    print(in_puzzle_horizontal(puzzle, name))

do_tasks(PUZZLE1, 'brian')

# *task* 2: call do_tasks on PUZZLE1 and 'nick'.
# Your code should work on 'nick' with no other changes made.
# If it doesn't work, check your code in do_tasks.
# Hint: you shouldn't be using 'brian' anywhere in do_tasks.
do_tasks(PUZZLE1, 'nick')
# *task* 7: call do_tasks on PUZZLE2 (that's a 2!) and 'nick'.
# Your code should work on the bigger puzzle with no changes made to do_tasks.
# If it doesn't work properly, go over your code carefully and fix it.
do_tasks(PUZZLE2, 'nick')
# *task* 9b: print the results of calling in_puzzle on PUZZLE1 and 'nick'.
# Add only one line below. Your code should print only True or False.
print(in_puzzle(PUZZLE1, 'nick'))

# *task* 9c: print the results of calling in_puzzle on PUZZLE2 and 'anya'.
# Add only one line below. Your code should print only True or False.
print(in_puzzle(PUZZLE2, 'anya'))
