from collections import defaultdict, deque

# Priority Queue for the open list
class PQ:
    def __init__(self):
        self.data = [None] * 5001
        self.data[4999] = 0
        self.data[5000] = 100000

    def add(self, my_path, new_cost):
        self.data[4999] += 1
        self.data[5000] = min(self.data[5000], new_cost)
        if not self.data[new_cost]:
            self.data[new_cost] = []
        self.data[new_cost].append(my_path)

    def lowest_cost(self, start):
        if self.data[4999] == 0:
            return 100000
        else:
            i = start
            while not self.data[i]:
                i += 1
            return i

    def remove(self):
        old_cost = self.data[5000]
        if self.data[4999] == 0:
            return None
        ans = self.data[old_cost].pop(0)
        self.data[4999] -= 1
        self.data[5000] = self.lowest_cost(old_cost)
        return ans

    def peek(self):
        old_cost = self.data[5000]
        return None if self.data[4999] == 0 else self.data[old_cost][0]

    def empty(self):
        return self.data[4999] == 0

# Hash table for the closed list
class MyHT:
    def __init__(self, size):
        self.size = size
        self.data = defaultdict(list)

    def hash_fn(self, s):
        def hash_helper(start, l):
            return start if not l else hash_helper(l[0] + start * 65559, l[1:])
        
        ans = 0
        for item in s:
            ans = hash_helper(ans, item)
        return ans

    def remove(self, key):
        hash_val = self.hash_fn(key) % self.size
        self.data[hash_val] = [kv for kv in self.data[hash_val] if kv[0] != key]

    def add(self, key, value):
        hash_val = self.hash_fn(key) % self.size
        self.data[hash_val].append((key, value))

    def get(self, key):
        hash_val = self.hash_fn(key) % self.size
        for k, v in self.data[hash_val]:
            if k == key:
                return v
        return None

# A* implementation
class MyPath:
    def __init__(self, state, previous=None, cost_so_far=0, total_cost=0):
        self.state = state
        self.previous = previous
        self.cost_so_far = cost_so_far
        self.total_cost = total_cost

    def states(self):
        if not self:
            return []
        return [self.state] + MyPath.states(self.previous)

expanded = 0
generated = 1

def astar(start_state, goal_p, successors, cost_fn, remaining_cost_fn):
    global expanded, generated
    expanded, generated = 0, 1
    open_list = PQ()
    closed_list = MyHT(1000000)
    open_list.add(MyPath(start_state), 0)
    while not open_list.empty():
        if goal_p(open_list.peek().state):
            return list(reversed(MyPath.states(open_list.peek())))
        my_path = open_list.remove()
        state = my_path.state
        new_val = my_path.total_cost
        hash_val = closed_list.hash_fn(state)
        closed_val = closed_list.get(state)
        if not closed_val or new_val < closed_val:
            expanded += 1
            if closed_val:
                closed_list.remove(state)
            closed_list.add(state, new_val)
            for state2 in successors(state):
                cost = my_path.cost_so_far + cost_fn(state, state2)
                cost2 = remaining_cost_fn(state2)
                total_cost = cost + cost2
                generated += 1
                open_list.add(MyPath(state2, my_path, cost, total_cost), total_cost)

# Constants for the game
EMPTY = 0
WALL = 1
KEEPER = 2
BOX = 3
GOAL = 4
BOX_ON_GOAL = 5
KEEPER_ON_GOAL = 6

# Function to get the content of a square
def get_square(S, r, c):
    """
    Get the integer content of state S at square (r,c).
    If the square is outside the scope of the problem, return WALL.
    """
    # Get the dimensions of the state
    if not S:
        return WALL
    
    rows = len(S)
    if rows == 0:
        return WALL
    
    cols = len(S[0])
    
    # Check if the coordinates are within bounds
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return WALL
    
    return S[r][c]

# Function to set the content of a square
def set_square(S, r, c, v):
    """
    Return a new state S' that is obtained by setting square (r,c) to value v.
    This function should not modify the input state.
    """
    # Create a copy of the state to avoid modifying the input
    S_prime = [row[:] for row in S]
    
    # Set the value at the specified position
    S_prime[r][c] = v
    
    return S_prime

# Function to try a move
def try_move(S, D):
    """
    Return the state that is the result of moving the keeper in state S in direction D.
    Return NIL if the move is invalid (e.g. there is a wall in that direction).
    D can be 'up', 'down', 'left', or 'right'.
    """
    # Find the keeper's position
    keeper_row, keeper_col = None, None
    for r in range(len(S)):
        for c in range(len(S[0])):
            if S[r][c] == KEEPER or S[r][c] == KEEPER_ON_GOAL:
                keeper_row, keeper_col = r, c
                break
        if keeper_row is not None:
            break
    
    # Determine the new position based on the direction
    new_row, new_col = keeper_row, keeper_col
    if D == 'up':
        new_row -= 1
    elif D == 'down':
        new_row += 1
    elif D == 'left':
        new_col -= 1
    elif D == 'right':
        new_col += 1
    else:
        return None  # Invalid direction
    
    # Check what's in the new position
    content = get_square(S, new_row, new_col)
    
    # If the new position is a wall, the move is invalid
    if content == WALL:
        return None
    
    # If the new position contains a box, check if it can be pushed
    if content == BOX or content == BOX_ON_GOAL:
        # Calculate where the box would end up
        box_new_row, box_new_col = new_row, new_col
        if D == 'up':
            box_new_row -= 1
        elif D == 'down':
            box_new_row += 1
        elif D == 'left':
            box_new_col -= 1
        elif D == 'right':
            box_new_col += 1
        
        # Check what's in the box's new position
        box_new_content = get_square(S, box_new_row, box_new_col)
        
        # If the box's new position is a wall or another box, the move is invalid
        if box_new_content == WALL or box_new_content == BOX or box_new_content == BOX_ON_GOAL:
            return None
        
        # Create a new state with the updated positions
        new_state = [row[:] for row in S]
        
        # Update the box's position
        if box_new_content == EMPTY:
            new_state[box_new_row][box_new_col] = BOX
        elif box_new_content == GOAL:
            new_state[box_new_row][box_new_col] = BOX_ON_GOAL
        
        # Update the keeper's new position (where the box was)
        if content == BOX:
            new_state[new_row][new_col] = KEEPER
        elif content == BOX_ON_GOAL:
            new_state[new_row][new_col] = KEEPER_ON_GOAL
        
        # Update the keeper's old position
        if S[keeper_row][keeper_col] == KEEPER:
            new_state[keeper_row][keeper_col] = EMPTY
        elif S[keeper_row][keeper_col] == KEEPER_ON_GOAL:
            new_state[keeper_row][keeper_col] = GOAL
        
        return new_state
    
    # If the new position is empty or a goal, just move the keeper
    new_state = [row[:] for row in S]
    
    # Update the keeper's new position
    if content == EMPTY:
        new_state[new_row][new_col] = KEEPER
    elif content == GOAL:
        new_state[new_row][new_col] = KEEPER_ON_GOAL
    
    # Update the keeper's old position
    if S[keeper_row][keeper_col] == KEEPER:
        new_state[keeper_row][keeper_col] = EMPTY
    elif S[keeper_row][keeper_col] == KEEPER_ON_GOAL:
        new_state[keeper_row][keeper_col] = GOAL
    
    return new_state

# Function to check if a state is a goal state
def goal_test(state):
    """
    Returns true if and only if the state is a goal state of the game.
    A state is a goal state if it satisfies the game terminating condition.
    """
    # A state is a goal state if all boxes are on goal positions
    for row in state:
        for square in row:
            # If there's a box that's not on a goal, it's not a goal state
            if square == BOX:
                return False
    return True

# Function to generate next states
def next_states(state):
    """
    Returns the list of all states that can be reached from the given state in one move.
    """
    possible_states = []
    
    # Try each possible direction
    for direction in ['up', 'down', 'left', 'right']:
        new_state = try_move(state, direction)
        if new_state:  # If the move is valid
            possible_states.append(new_state)
    
    return possible_states

# Trivial admissible heuristic function
def h0(state):
    """
    A trivial admissible heuristic that returns the constant 0.
    """
    return 0

# Heuristic based on number of boxes not on goals
def h1(state):
    """
    A heuristic function that returns the number of boxes which are not on goal positions.
    This heuristic is admissible because each box not on a goal requires at least one move to reach a goal.
    """
    count = 0
    for row in state:
        for square in row:
            if square == BOX:
                count += 1
    return count

# Custom heuristic function - you should replace this with your own ID
def h123456789(state):
    """
    A custom heuristic function that aims to make the A* search more efficient.
    This heuristic is admissible because it never overestimates the cost to reach the goal.
    """
    # This is a more sophisticated heuristic that calculates the Manhattan distance
    # from each box to its nearest goal, and sums these distances
    
    # First, find all boxes and goals
    boxes = []
    goals = []
    
    for r in range(len(state)):
        for c in range(len(state[0])):
            if state[r][c] == BOX:
                boxes.append((r, c))
            elif state[r][c] == GOAL or state[r][c] == BOX_ON_GOAL or state[r][c] == KEEPER_ON_GOAL:
                goals.append((r, c))
    
    # If there are no boxes not on goals, return 0
    if not boxes:
        return 0
    
    # Calculate the Manhattan distance from each box to its nearest goal
    total_distance = 0
    for box_r, box_c in boxes:
        min_dist = float('inf')
        for goal_r, goal_c in goals:
            dist = abs(box_r - goal_r) + abs(box_c - goal_c)
            min_dist = min(min_dist, dist)
        total_distance += min_dist
    
    return total_distance

# Helper function to calculate the cost of moving from one state to another
def move_cost(state1, state2):
    """
    Returns the cost of moving from state1 to state2.
    In this game, each move has a cost of 1.
    """
    return 1

# This part would include test code and examples of using the functions

# Test the implementation with the provided examples
if __name__ == "__main__":
    # Example Sokoban states
    # 0 = empty, 1 = wall, 2 = keeper, 3 = box, 4 = goal, 5 = box on goal, 6 = keeper on goal
    # From the PDF examples
    s1 = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 4, 1],
        [1, 0, 2, 0, 1],
        [1, 0, 3, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    
    # PDF example state with the box to the right of the keeper
    s0 = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 4, 1],
        [1, 0, 2, 3, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    s2 = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 4, 1],
        [1, 0, 2, 3, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    
    # Create test states for box pushing in all four directions
    # Box above keeper with space for pushing
    push_up = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 3, 0, 0, 1],
        [1, 0, 0, 2, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    
    # Box below keeper with space for pushing
    push_down = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 2, 0, 0, 1],
        [1, 0, 0, 3, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    
    # Box to the left of keeper with space for pushing
    push_left = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 3, 2, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    
    # Box to the right of keeper with space for pushing
    push_right = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 2, 3, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    
    # Test box pushing in all directions
    print("Testing box pushing:")
    
    print("\nPush Up Test:")
    print("Initial state:")
    for row in push_up:
        print(row)
    up_move = try_move(push_up, 'up')
    if up_move:
        print("Result after pushing up:")
        for row in up_move:
            print(row)
    else:
        print("Cannot push up")
        
    print("\nPush Down Test:")
    print("Initial state:")
    for row in push_down:
        print(row)
    down_move = try_move(push_down, 'down')
    if down_move:
        print("Result after pushing down:")
        for row in down_move:
            print(row)
    else:
        print("Cannot push down")
        
    print("\nPush Left Test:")
    print("Initial state:")
    for row in push_left:
        print(row)
    left_move = try_move(push_left, 'left')
    if left_move:
        print("Result after pushing left:")
        for row in left_move:
            print(row)
    else:
        print("Cannot push left")
        
    print("\nPush Right Test:")
    print("Initial state:")
    for row in push_right:
        print(row)
    right_move = try_move(push_right, 'right')
    if right_move:
        print("Result after pushing right:")
        for row in right_move:
            print(row)
    else:
        print("Cannot push right")
    
    # Continue with original tests...
    print("\n\nExample 0 (PDF state):")
    print("Initial state s0:")
    for row in s0:
        print(row)
    
    print("\nIs this a goal state?", goal_test(s0))
    
    print("\nPossible next states from s0:")
    next_states_list = next_states(s0)
    for i, state in enumerate(next_states_list):
        print(f"Next state {i+1}:")
        for row in state:
            print(row)
        print()
    
    print("h0 heuristic:", h0(s0))
    print("h1 heuristic:", h1(s0))
    print("Custom heuristic:", h123456789(s0))
    
    # Test s1
    print("\n\nExample 1:")
    print("Initial state s1:")
    for row in s1:
        print(row)
    
    print("\nIs this a goal state?", goal_test(s1))
    
    print("\nPossible next states from s1:")
    next_states_list = next_states(s1)
    for i, state in enumerate(next_states_list):
        print(f"Next state {i+1}:")
        for row in state:
            print(row)
        print()
    
    print("h0 heuristic:", h0(s1))
    print("h1 heuristic:", h1(s1))
    print("Custom heuristic:", h123456789(s1))
    
    # Test s2
    print("\n\nExample 2:")
    print("Initial state s2:")
    for row in s2:
        print(row)
    
    print("\nIs this a goal state?", goal_test(s2))
    
    print("\nPossible next states from s2:")
    next_states_list = next_states(s2)
    for i, state in enumerate(next_states_list):
        print(f"Next state {i+1}:")
        for row in state:
            print(row)
        print()
    
    print("h0 heuristic:", h0(s2))
    print("h1 heuristic:", h1(s2))
    print("Custom heuristic:", h123456789(s2))
    
    # Test A* search with s1
    print("\nRunning A* search on s1 with h1 heuristic...")
    path = astar(s1, goal_test, next_states, move_cost, h1)
    
    if path:
        print(f"Solution found with {len(path)-1} moves:")
        for i, state in enumerate(path):
            print(f"Step {i}:")
            for row in state:
                print(row)
            print()
    else:
        print("No solution found for s1.")
