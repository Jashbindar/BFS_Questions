# LAB 1 (AI)


# %% [markdown]
# Breadth-First Search [BFS]

# %%
from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': ['G'],
    'E': ['G'],
    'F': ['G'],
    'G': []
}

def dfs_stack(start, end, arg_graph):
    visited = set()
    queue = deque([start])
    parent = {}

    while queue:
        curr_node = queue.popleft()
        if curr_node == end:
            path = []
            while curr_node != start:
                path.append(curr_node)
                curr_node = parent[curr_node]
            path.reverse()
            return path
        visited.add(curr_node)
        for neighbour in arg_graph[curr_node]:
            if neighbour not in queue:
                parent[neighbour] = curr_node
                queue.append(neighbour)



    return None

# output // result code
start = 'A'
end = 'G'

path_created = dfs_stack(start, end, graph)

if path_created is None:
    print("Path not found.")
else:
    print(f"{start} -> ", end="")
    print(" -> " .join(path_created))

# %%
from collections import deque

maze = [
    ['S', '.', '.', '#', 'G'],
    ['#', '#', '.', '#', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '#', '#', '#', '.'],
    ['.', '.', '.', '.', '.']
]

def goBottom(y, x):
    x = x
    y = y + 1
    pos_at = (y, x)
    return pos_at

def goTop(y, x):
    x = x
    y = y - 1
    pos_at = (y, x)
    return pos_at

def goRight(y, x):
    x = x + 1
    y = y
    pos_at = (y, x)
    return pos_at

def goLeft(y, x):
    x = x - 1
    y = y
    pos_at = (y, x)
    return pos_at

def getNeighbor(curr_pos, arg_bfs_queue, arg_maze, arg_visited, arg_parent):
    planned_path = curr_pos
    movement_direction = [goBottom, goTop, goRight, goLeft]

    for movement in movement_direction:
        y, x = planned_path
        tmp_curr_pos = movement(y, x)

        # returns a tuple
        if (tmp_curr_pos[0] >= 0 and tmp_curr_pos[0] <= len(arg_maze) - 1) and (tmp_curr_pos[1] >= 0 and tmp_curr_pos[1] <= len(arg_maze[0]) - 1): 
            # check if its out of list
            if tmp_curr_pos not in arg_bfs_queue and tmp_curr_pos not in arg_visited and (arg_maze[tmp_curr_pos[0]][tmp_curr_pos[1]] == '.' or arg_maze[tmp_curr_pos[0]][tmp_curr_pos[1]] == 'G'):
                # queues the cooridinates in the form of tuple (y,x)
                arg_bfs_queue.append(tmp_curr_pos)
                arg_parent[tmp_curr_pos] = curr_pos
    

def BFS(arg_start, arg_goal, arg_maze):
    start, end = None, None
    visited = set()
    parent = {}

    for y in range (len(maze)):
        for x in range (len(maze[y])):
            if maze[y][x] == arg_start:
                start = (y, x)
            elif maze [y][x] == arg_goal:
                end = (y, x)

    bfs_queue = deque([start])
    while bfs_queue:
        curr_node = bfs_queue.popleft()

        if curr_node == end:
            path = []
            while curr_node != start:
                path.append(curr_node)
                curr_node = parent[curr_node]
            path.reverse()
            return path
        visited.add(curr_node)
        getNeighbor(curr_node, bfs_queue, maze, visited, parent)
    return None

# output // result code
start = 'S'
goal = 'G'

for y in range (len(maze)):
    for x in range (len(maze[y])):
        if maze[y][x] == start:
            start_pos = (y, x)

path_built = BFS(start, goal, maze)


if path_built is None:
    print("No path found")
else:
    print(f'{start_pos} -> ', end="")
    print(" -> ".join(str(pos) for pos in path_built))

# %%
from collections import deque

maze = [
    ['S', '.', '.', '#', 'G'],
    ['#', '#', '.', '#', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '#', '#', '#', '.'],
    ['.', '.', '.', '.', '.']
]

def goBottom(y, x):
    x = x
    y = y + 1
    pos_at = (y, x)
    return pos_at

def goTop(y, x):
    x = x
    y = y - 1
    pos_at = (y, x)
    return pos_at

def goRight(y, x):
    x = x + 1
    y = y
    pos_at = (y, x)
    return pos_at

def goLeft(y, x):
    x = x - 1
    y = y
    pos_at = (y, x)
    return pos_at

def getNeighbor(curr_pos, arg_bfs_queue, arg_maze, arg_visited, arg_parent):
    planned_path = curr_pos
    movement_direction = [goBottom, goTop, goRight, goLeft]

    for movement in movement_direction:
        y, x = planned_path
        tmp_curr_pos = movement(y, x)

        # returns a tuple
        if (tmp_curr_pos[0] >= 0 and tmp_curr_pos[0] <= len(arg_maze) - 1) and (tmp_curr_pos[1] >= 0 and tmp_curr_pos[1] <= len(arg_maze[0]) - 1): 
            # check if its out of list
            if tmp_curr_pos not in arg_bfs_queue and tmp_curr_pos not in arg_visited and (arg_maze[tmp_curr_pos[0]][tmp_curr_pos[1]] == '.' or arg_maze[tmp_curr_pos[0]][tmp_curr_pos[1]] == 'G'):
                # queues the cooridinates in the form of tuple (y,x)
                arg_bfs_queue.append(tmp_curr_pos)
                arg_parent[tmp_curr_pos] = curr_pos
    

def BFS(arg_start, arg_goal, arg_maze):
    start, end = None, None
    visited = set()
    parent = {}

    for y in range (len(maze)):
        for x in range (len(maze[y])):
            if maze[y][x] == arg_start:
                start = (y, x)
            elif maze [y][x] == arg_goal:
                end = (y, x)

    bfs_queue = deque([start])
    while bfs_queue:
        curr_node = bfs_queue.popleft()

        if curr_node == end:
            path = []
            while curr_node != start:
                path.append(curr_node)
                curr_node = parent[curr_node]
            path.reverse()
            return path
        visited.add(curr_node)
        getNeighbor(curr_node, bfs_queue, maze, visited, parent)
    return None

# output // result code
start = 'S'
goal = 'G'

for y in range (len(maze)):
    for x in range (len(maze[y])):
        if maze[y][x] == start:
            start_pos = (y, x)

path_built = BFS(start, goal, maze)


if path_built is None:
    print("No path found")
else:
    print(f'{start_pos} -> ', end="")
    print(" -> ".join(str(pos) for pos in path_built))

# %%
from collections import deque

def dfs_stack (start, end, word_list):
    word_set = set(word_list)
    queue = deque([start])
    visited = set()
    parent = {}

    while queue:
        curr_word = queue.popleft()
        if curr_word == end:
            path = []
            while curr_word != start:
                path.append(curr_word)
                curr_word = parent[curr_word]
            path.append(start)
            path.reverse()
            return path
        
        for word in word_set:
            if word not in visited:
                # diff is used because the algorithm will instantly take the first letter with difference
                # since 'cog' has 3 different letters, it will skip the other words and convert 'hit' to 'cog' immediately
                diff = 0 # check the letter difference
                for i in range(len(word)):
                    if curr_word[i] != word[i]:
                        diff += 1
                    if diff > 1:
                        break
                if diff == 1:
                    queue.append(word)
                    visited.add(word)
                    parent[word] = curr_word

    return None

word_list = ["hit", "hot", "dot", "dog", "lot", "log", "cog"]

start = 'hit'
end = 'cog'
path_built = dfs_stack(start, end, word_list)

if path_built is None:
    print("No path found")
else:
    print(f"{start} -> ", end=" ")
    print(" -> " .join(path_built))

# %%
from collections import deque

# used to initialize the nodes
class Node:
    def __init__(self, value):

        #initialize the parent root
        self.value = value

        #initializes the child roots
        self.left = None
        self.right = None

def dfs_stack(root):
    visited = []
    queue = deque([root])

    if not root:
        return []
    
    while queue:
        node = queue.popleft()
        visited.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return visited

# output
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)

print(dfs_stack(root))

# %% [markdown]
# Depth-First Search [DFS]

# %%
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': ['G'],
    'E': ['G'],
    'F': ['G'],
    'G': []
}

def dfs_stack(start, goal, graph):
    stack = [start]
    visited = set()
    parent = {start: None}

    while stack:
        curr = stack.pop()
        print(curr)

        if curr == goal:
            path = []
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return path
        
        if curr not in visited:
            visited.add(curr)
            for neighbour in reversed(graph[curr]):
                print(neighbour)
                if neighbour not in visited:
                    stack.append(neighbour)
                    parent[neighbour] = curr
                    
                             
    return None

print(dfs_stack('A', 'G', graph))

# %%
maze = [
    ['S', '.', '.', '#', 'G'],
    ['#', '#', '.', '#', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '#', '#', '#', '.'],
    ['.', '.', '.', '.', '.']
]


def goBottom(y, x):
    x = x
    y = y + 1
    pos_at = (y, x)
    return pos_at

def goTop(y, x):
    x = x
    y = y - 1
    pos_at = (y, x)
    return pos_at

def goRight(y, x):
    x = x + 1
    y = y
    pos_at = (y, x)
    return pos_at

def goLeft(y, x):
    x = x - 1
    y = y
    pos_at = (y, x)
    return pos_at

def getNeighbor(curr_pos, arg_dfs_stack, arg_maze, arg_visited, arg_parent):
    movement_direction = [goBottom, goTop, goRight, goLeft]

    for move in movement_direction:
        y,x = curr_pos
        ny,nx = move(y,x)

        if 0 <= ny < len(arg_maze) and 0 <= nx < len(arg_maze[0]):
            if (ny, nx) not in arg_visited:
                if arg_maze[ny][nx] == '.' or arg_maze[ny][nx] == 'G':
                    arg_dfs_stack.append((ny,nx))
                    arg_parent[(ny,nx)] = curr_pos

def DFS(arg_start, arg_goal, arg_maze):
    start, end = None, None
    visited = set()
    parent = {}

    for y in range (len(arg_maze)):
        for x in range (len(arg_maze[y])):
            if arg_maze[y][x] == arg_start:
                start = (y, x)
            elif arg_maze [y][x] == arg_goal:
                end = (y, x)

    stack = [start]
    while stack:
        curr_node = stack.pop()

        if curr_node == end:
            path = []
            while curr_node != start:
                path.append(curr_node)
                curr_node = parent[curr_node]
            path.reverse()
            return path
        
        if curr_node not in visited:
            visited.add(curr_node)
            getNeighbor(curr_node, stack, arg_maze, visited, parent)

    return None

# output // result code
start = 'S'
goal =  'G'

for y in range (len(maze)):
    for x in range (len(maze[y])):
        if maze[y][x] == start:
            start_pos = (y, x)

path_built = DFS(start, goal, maze)

if path_built is None:
    print("No path found")
else:
    print(f'{start_pos} -> ', end="")
    print(" -> ".join(str(pos) for pos in path_built))

# %%
def dfs_stack (start, end, word_list):
    word_set = set(word_list)
    stack = [start]
    visited = set()
    parent = {}

    while stack:
        curr_word = stack.pop()
        if curr_word == end:
            path = []
            while curr_word != start:
                path.append(curr_word)
                curr_word = parent[curr_word]
            path.append(start)
            path.reverse()
            return path
        
        for word in word_set:
            if word not in visited:
                # diff is used because the algorithm will instantly take the first letter with difference
                # since 'cog' has 3 different letters, it will skip the other words and convert 'hit' to 'cog' immediately
                diff = 0 # check the letter difference
                for i in range(len(word)):
                    if curr_word[i] != word[i]:
                        diff += 1
                    if diff > 1:
                        break
                if diff == 1:
                    stack.append(word)
                    visited.add(word)
                    parent[word] = curr_word

    return None

word_list = ["hit", "hot", "dot", "dog", "lot", "log", "cog"]

start = 'hit'
end = 'cog'
path_built = dfs_stack(start, end, word_list)

if path_built is None:
    print("No path found")
else:
    print(" -> " .join(path_built))

# %%
class Node:
    def __init__(self, Value):
        self.Value = Value
        self.Left = None
        self.Right = None

def dfs_stack(node):
    visited = []
    stack = [node]
    
    if not node:
        return []
        
    while stack:
        root = stack.pop()
        visited.append(root.Value)
        if root.Right:
            stack.append(root.Right)
        if root.Left:
            stack.append(root.Left)

    return visited

# output // result

root = Node(1)
root.Left = Node(2)
root.Right = Node(3)
root.Left.Left = Node(4)
root.Left.Right = Node(5)
root.Right.Left = Node(6)
root.Right.Right = Node(7)

print(dfs_stack(root))


# %% [markdown]
# Uniform Cost Search (UCS)

# %%
#   what is "heapq"
#   -> "heapq" is a priority queue data structure which UCS also uses
import heapq


graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 3), ('E', 2)],
    'C': [('F', 5)],
    'D': [('G', 4)],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}

def ucs(start, goal, graph):
    queue = [(0, start, [start])]
    visited = set()
    parent = {start: None}

    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == goal:
            return path, cost
        visited.add(node)
        for neighbour, weightage in graph[node]:
            if neighbour not in visited:
                parent[neighbour] = node
                # "cost + weightage" adds the weightage of the current node with the total weightage of the previous paths
                heapq.heappush(queue, (cost + weightage, neighbour, path + [neighbour]))

    return None, None


# output // result code
start = 'A'
end = 'G'

path_created, total_cost = ucs(start, end, graph)

if path_created is None:
    print("No path found")
else:
    print( " -> " .join(path_created))
    print(f"Total Cost (weightage): {total_cost}")

# %%
import heapq

maze = [
    [1, 1, 1, 99, 1],
    [99, 99, 1, 99, 1],
    [1, 1, 1, 1, 1],
    [1, 99, 99, 99, 1],
    [1, 1, 1, 1, 1]
]

def goBottom(y, x):
    x = x
    y = y + 1
    pos_at = (y, x)
    return pos_at

def goTop(y, x):
    x = x
    y = y - 1
    pos_at = (y, x)
    return pos_at

def goRight(y, x):
    x = x + 1
    y = y
    pos_at = (y, x)
    return pos_at

def goLeft(y, x):
    x = x - 1
    y = y
    pos_at = (y, x)
    return pos_at

def getNeighbor(curr_pos, arg_ucs_queue, arg_maze, arg_visited, arg_parent, cost, path):
    movement_direction = [goBottom, goTop, goRight, goLeft]

    for move in movement_direction:
        y,x = curr_pos
        ny,nx = move(y,x)

        if 0 <= ny < len(arg_maze) and 0 <= nx < len(arg_maze[0]):
            if (ny, nx) not in arg_visited:
                if arg_maze[ny][nx] != 99:
                    new_cost = cost + arg_maze[ny][nx]
                    heapq.heappush(arg_ucs_queue, (new_cost, (ny, nx), path + [(ny, nx)]))
                    arg_parent[(ny,nx)] = curr_pos
    

def BFS(arg_start, arg_goal, arg_maze):
    start, end = None, None
    visited = set()
    parent = {}

    start = arg_start
    end = arg_goal

    ucs_queue = [(0, start, [start])]

    while ucs_queue:
        cost, curr_pos, path = heapq.heappop(ucs_queue)

        if curr_pos == end:
            return cost, path
            
        if curr_pos not in visited:
            visited.add(curr_pos)
            getNeighbor(curr_pos, ucs_queue, maze, visited, parent, cost, path)

    return None

# output // result code
start = (0,0)
goal = (0,4)

total_cost, path_built = BFS(start, goal, maze)

if path_built is None:
    print("No path found")
else:
    print(f'{(start)} -> ', end="")
    print(" -> ".join(str(pos) for pos in path_built))
    print(f"Total Cost (weightage): {total_cost}")

# %%
import heapq

graph = {
    'A': [('B', 2), ('C', 5)],
    'B': [('A', 2), ('D', 1)],
    'C': [('A', 5), ('D', 2)],
    'D': [('B', 1), ('C', 2), ('G', 3)],
    'G': []
}

def ucs(start, goal, graph):
    queue = [(0, start, [start])]
    visited = set()
    parent = {start: None}

    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == goal:
            return path, cost
        visited.add(node)
        for neighbour, weightage in graph[node]:
            if neighbour not in visited:
                parent[neighbour] = node

                # "cost + weightage" adds the weightage of the current node with the total weightage of the previous paths
                heapq.heappush(queue, (cost + weightage, neighbour, path + [neighbour]))

    return None


# output // result code
start = 'A'
end = 'G'

# path_created = list, total_cost = int
path_created, total_cost = ucs(start, end, graph)

if path_created is None:
    print("No path found")
else:
    print( " -> " .join(path_created))
    print(f"Total Cost (weightage): {total_cost}")