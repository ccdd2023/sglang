code = '''"""Large multi-file codebase for KV reuse experiments."""

DATA_STRUCTURES = """
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None
    def _height(self, node):
        return node.height if node else 0
    def _balance_factor(self, node):
        return self._height(node.left) - self._height(node.right)
    def _update_height(self, node):
        node.height = 1 + max(self._height(node.left), self._height(node.right))
    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self._update_height(y)
        self._update_height(x)
        return x
    def _rotate_left(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self._update_height(x)
        self._update_height(y)
        return y
    def insert(self, key):
        self.root = self._insert(self.root, key)
    def _insert(self, node, key):
        if not node:
            return AVLNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        else:
            node.right = self._insert(node.right, key)
        self._update_height(node)
        balance = self._balance_factor(node)
        if balance > 1 and key < node.left.key:
            return self._rotate_right(node)
        if balance < -1 and key > node.right.key:
            return self._rotate_left(node)
        if balance > 1 and key > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1 and key < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

class MinHeap:
    def __init__(self):
        self.heap = []
    def parent(self, i):
        return (i - 1) // 2
    def left_child(self, i):
        return 2 * i + 1
    def right_child(self, i):
        return 2 * i + 2
    def insert(self, key):
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)
    def _heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    def extract_min(self):
        if not self.heap:
            return None
        root = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self._heapify_down(0)
        return root
    def _heapify_down(self, i):
        min_idx = i
        left = self.left_child(i)
        right = self.right_child(i)
        if left < len(self.heap) and self.heap[left] < self.heap[min_idx]:
            min_idx = left
        if right < len(self.heap) and self.heap[right] < self.heap[min_idx]:
            min_idx = right
        if min_idx != i:
            self.heap[i], self.heap[min_idx] = self.heap[min_idx], self.heap[i]
            self._heapify_down(min_idx)

class LinkedListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    def append(self, val):
        if not self.head:
            self.head = LinkedListNode(val)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = LinkedListNode(val)
    def find(self, val):
        curr = self.head
        while curr:
            if curr.val == val:
                return True
            curr = curr.next
        return False
    def delete(self, val):
        if not self.head:
            return
        if self.head.val == val:
            self.head = self.head.next
            return
        curr = self.head
        while curr.next and curr.next.val != val:
            curr = curr.next
        if curr.next:
            curr.next = curr.next.next
"""

GRAPH_ALGORITHMS = """
from collections import defaultdict, deque
import heapq

class Graph:
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed
    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))
    def dijkstra(self, start):
        dist = {start: 0}
        prev = {start: None}
        pq = [(0, start)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            for v, w in self.adj[u]:
                if v not in dist or dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
        return dist, prev
    def bfs(self, start):
        visited = {start}
        queue = deque([start])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in self.adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return order
    def dfs(self, start):
        visited = set()
        order = []
        def _dfs(u):
            visited.add(u)
            order.append(u)
            for v, _ in self.adj[u]:
                if v not in visited:
                    _dfs(v)
        _dfs(start)
        return order
    def topological_sort(self):
        in_degree = defaultdict(int)
        for u in self.adj:
            for v, _ in self.adj[u]:
                in_degree[v] += 1
        queue = deque([u for u in self.adj if in_degree[u] == 0])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in self.adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return order if len(order) == len(self.adj) else []
    def has_cycle(self):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        def _dfs(u):
            color[u] = GRAY
            for v, _ in self.adj[u]:
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and _dfs(v):
                    return True
            color[u] = BLACK
            return False
        for u in list(self.adj.keys()):
            if color[u] == WHITE and _dfs(u):
                return True
        return False
    def connected_components(self):
        visited = set()
        components = []
        def _dfs(u, comp):
            visited.add(u)
            comp.append(u)
            for v, _ in self.adj[u]:
                if v not in visited:
                    _dfs(v, comp)
        for u in list(self.adj.keys()):
            if u not in visited:
                comp = []
                _dfs(u, comp)
                components.append(comp)
        return components
"""

SORTING = """
class Sorter:
    def merge_sort(self, arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        return self._merge(left, right)
    def _merge(self, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    def quick_sort(self, arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return self.quick_sort(left) + middle + self.quick_sort(right)
    def heap_sort(self, arr):
        def heapify(a, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and a[left] > a[largest]:
                largest = left
            if right < n and a[right] > a[largest]:
                largest = right
            if largest != i:
                a[i], a[largest] = a[largest], a[i]
                heapify(a, n, largest)
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(arr, i, 0)
        return arr
    def insertion_sort(self, arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    def counting_sort(self, arr):
        if not arr:
            return arr
        min_val = min(arr)
        max_val = max(arr)
        count = [0] * (max_val - min_val + 1)
        for x in arr:
            count[x - min_val] += 1
        result = []
        for i, c in enumerate(count):
            result.extend([i + min_val] * c)
        return result
"""

def get_full_codebase():
    return "# File 1: data_structures.py\\n" + DATA_STRUCTURES + "\\n# File 2: graph_algorithms.py\\n" + GRAPH_ALGORITHMS + "\\n# File 3: sorting.py\\n" + SORTING
'''

with open('/home/gfy/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow/large_codebase.py', 'w') as f:
    f.write(code)
print("Created large_codebase.py")
