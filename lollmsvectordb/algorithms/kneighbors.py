import numpy as np
import heapq
from collections import defaultdict

class NearestNeighbors:
    def __init__(self, n_neighbors=5, algorithm='auto', metric='euclidean', leaf_size=30):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.X = None
        self.tree = None
        self.hnsw_graph = None

    def fit(self, X):
        self.X = np.array(X)
        if self.algorithm == 'auto':
            if self.X.shape[1] < 3:
                self.algorithm = 'kd_tree'
            elif self.X.shape[1] < 20:
                self.algorithm = 'ball_tree'
            else:
                self.algorithm = 'brute'
        
        if self.algorithm == 'kd_tree':
            self.tree = self._build_kd_tree(self.X, depth=0)
        elif self.algorithm == 'ball_tree':
            self.tree = self._build_ball_tree(self.X)
        elif self.algorithm == 'hnsw':
            self.hnsw_graph = self._build_hnsw_graph()
        return self

    def _build_kd_tree(self, X, depth):
        if len(X) <= self.leaf_size:
            return {'point': X, 'left': None, 'right': None}
        
        axis = depth % X.shape[1]
        sorted_X = X[X[:, axis].argsort()]
        median = len(X) // 2

        return {
            'point': sorted_X[median],
            'left': self._build_kd_tree(sorted_X[:median], depth + 1),
            'right': self._build_kd_tree(sorted_X[median + 1:], depth + 1)
        }

    def _build_ball_tree(self, X):
        if len(X) <= self.leaf_size:
            return {'center': np.mean(X, axis=0), 'radius': self._max_distance(X), 'points': X}
        
        axis = np.argmax(np.var(X, axis=0))
        median = np.median(X[:, axis])
        left = X[X[:, axis] < median]
        right = X[X[:, axis] >= median]

        return {
            'center': np.mean(X, axis=0),
            'radius': self._max_distance(X),
            'left': self._build_ball_tree(left),
            'right': self._build_ball_tree(right)
        }

    def _build_hnsw_graph(self, M=5, ef_construction=100):
        n_samples = len(self.X)
        graph = defaultdict(set)
        
        for i in range(n_samples):
            if i == 0:
                continue
            
            curr_point = self.X[i]
            curr_neighbors = set()
            
            # Find ef_construction nearest neighbors
            candidates = [(self._distance(curr_point, self.X[j]), j) for j in range(i)]
            candidates.sort()
            
            for _, neighbor in candidates[:ef_construction]:
                curr_neighbors.add(neighbor)
                if len(curr_neighbors) >= M and len(graph[neighbor]) >= M:
                    break
            
            # Add edges
            for neighbor in curr_neighbors:
                graph[i].add(neighbor)
                graph[neighbor].add(i)
            
            # Ensure each node has at most M neighbors
            if len(graph[i]) > M:
                graph[i] = set(sorted(graph[i], key=lambda x: self._distance(curr_point, self.X[x]))[:M])
        
        return graph

    def _max_distance(self, X):
        center = np.mean(X, axis=0)
        return np.max(np.sqrt(np.sum((X - center)**2, axis=1)))

    def _distance(self, x, y):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x - y)**2))
        elif self.metric == 'cosine':
            return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        else:
            raise ValueError("Only 'euclidean' and 'cosine' metrics are implemented.")

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is None:
            X = self.X
        else:
            X = np.array(X)
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if self.algorithm == 'brute':
            return self._kneighbors_brute(X, n_neighbors, return_distance)
        elif self.algorithm in ['kd_tree', 'ball_tree']:
            return self._kneighbors_tree(X, n_neighbors, return_distance)
        elif self.algorithm == 'hnsw':
            return self._kneighbors_hnsw(X, n_neighbors, return_distance)

    def _kneighbors_brute(self, X, n_neighbors, return_distance):
        distances = np.array([[self._distance(x, y) for y in self.X] for x in X])
        indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        
        if return_distance:
            distances = np.sort(distances, axis=1)[:, :n_neighbors]
            return distances, indices
        else:
            return indices

    def _kneighbors_tree(self, X, n_neighbors, return_distance):
        def search_tree(node, point, heap):
            if node is None:
                return
            
            if 'points' in node:  # Leaf node
                for i, p in enumerate(node['points']):
                    dist = self._distance(point, p)
                    if len(heap) < n_neighbors or dist < -heap[0][0]:
                        if len(heap) == n_neighbors:
                            heapq.heappop(heap)
                        heapq.heappush(heap, (-dist, p))
            else:
                if self.algorithm == 'kd_tree':
                    axis = len(point) % point.shape[0]
                    if point[axis] < node['point'][axis]:
                        search_tree(node['left'], point, heap)
                        if len(heap) < n_neighbors or abs(point[axis] - node['point'][axis]) < -heap[0][0]:
                            search_tree(node['right'], point, heap)
                    else:
                        search_tree(node['right'], point, heap)
                        if len(heap) < n_neighbors or abs(point[axis] - node['point'][axis]) < -heap[0][0]:
                            search_tree(node['left'], point, heap)
                elif self.algorithm == 'ball_tree':
                    dist_to_center = self._distance(point, node['center'])
                    if len(heap) < n_neighbors or dist_to_center - node['radius'] < -heap[0][0]:
                        search_tree(node['left'], point, heap)
                        search_tree(node['right'], point, heap)

        results = []
        for point in X:
            heap = []
            search_tree(self.tree, point, heap)
            results.append(sorted([-d for d, p in heap]))
        
        distances = np.array(results)
        indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        
        if return_distance:
            return distances[:, :n_neighbors], indices
        else:
            return indices

    def _kneighbors_hnsw(self, X, n_neighbors, return_distance, ef_search=50):
        def search_hnsw(query, ef):
            entry_point = 0
            visited = set()
            candidates = [(self._distance(query, self.X[entry_point]), entry_point)]
            heapq.heapify(candidates)
            
            while candidates:
                _, current = heapq.heappop(candidates)
                if current in visited:
                    continue
                visited.add(current)
                
                for neighbor in self.hnsw_graph[current]:
                    if neighbor not in visited:
                        dist = self._distance(query, self.X[neighbor])
                        if len(candidates) < ef or dist < -candidates[0][0]:
                            heapq.heappush(candidates, (-dist, neighbor))
                        if len(candidates) > ef:
                            heapq.heappop(candidates)
            
            return [(-dist, idx) for dist, idx in candidates]

        results = []
        for point in X:
            neighbors = search_hnsw(point, ef_search)
            results.append(sorted(neighbors)[:n_neighbors])
        
        if return_distance:
            distances = np.array([[d for d, _ in r] for r in results])
            indices = np.array([[i for _, i in r] for r in results])
            return distances, indices
        else:
            indices = np.array([[i for _, i in r] for r in results])
            return indices

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        if X is None:
            X = self.X
        else:
            X = np.array(X)
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        n_samples_fit = self.X.shape[0]
        n_samples_X = X.shape[0]
        
        distances, indices = self.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)
        
        graph = np.zeros((n_samples_X, n_samples_fit), dtype=float)
        
        if mode == 'connectivity':
            for i in range(n_samples_X):
                graph[i, indices[i]] = 1
        elif mode == 'distance':
            for i in range(n_samples_X):
                graph[i, indices[i]] = distances[i]
        else:
            raise ValueError("Unsupported mode. Use 'connectivity' or 'distance'.")
        
        return graph
