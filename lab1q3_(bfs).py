from collections import deque

def bfs_queue (start, end, word_list):
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
path_built = bfs_queue(start, end, word_list)

if path_built is None:
    print(f"{start} -> ", end=" ")
else:
    print(" -> " .join(path_built))