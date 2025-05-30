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

def bfs_queue(start, end, arg_graph):
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
                print(neighbour)
                parent[neighbour] = curr_node
                queue.append(neighbour)

    return None

start = 'A'
end = 'G'
path_created = bfs_queue(start, end, graph)

if path_created is None:
    print("Path not found.")
else:
    print(f"{start} -> ", end="")
    print(" -> " .join(path_created))
