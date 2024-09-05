import numpy as np

class NearestNeighbors:
    def __init__(self, n_neighbors=5, algorithm='auto', metric='euclidean', leaf_size=30):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.X = None
        self.tree = None

    def fit(self, X):
        self.X = np.array(X)
        if self.algorithm == 'auto':
            if self.X.shape[1] < 3:
                self.algorithm = 'kd_tree'
            elif self.X.shape[1] < 20:
                self.algorithm = 'ball_tree'
            else:
                self.algorithm = 'brute'
        
        if self.algorithm in ['kd_tree', 'ball_tree']:
            self.tree = self._build_tree()
        return self

    def _build_tree(self):
        if self.algorithm == 'kd_tree':
            return self._build_kd_tree(self.X, depth=0)
        elif self.algorithm == 'ball_tree':
            return self._build_ball_tree(self.X)

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
        
        # Simple split on the dimension with largest variance
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

    def _max_distance(self, X):
        center = np.mean(X, axis=0)
        return np.max(np.sqrt(np.sum((X - center)**2, axis=1)))

    def cosine_distance(self, X, Y):
        X_norm = np.sqrt(np.sum(X**2, axis=1))[:, np.newaxis]
        Y_norm = np.sqrt(np.sum(Y**2, axis=1))[np.newaxis, :]
        cosine_sim = np.dot(X, Y.T) / (X_norm * Y_norm)
        return 1 - cosine_sim

    def euclidean_distance(self, X, Y):
        return np.sqrt(((X[:, np.newaxis, :] - Y[np.newaxis, :, :])**2).sum(axis=2))

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

    def _kneighbors_brute(self, X, n_neighbors, return_distance):
        if self.metric == 'cosine':
            distances = self.cosine_distance(X, self.X)
        elif self.metric == 'euclidean':
            distances = self.euclidean_distance(X, self.X)
        else:
            raise ValueError("Only 'cosine' and 'euclidean' metrics are implemented.")
        
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
                distances = self.euclidean_distance(point[np.newaxis, :], node['points']).flatten()
                for i, dist in enumerate(distances):
                    if len(heap) < n_neighbors or dist < -heap[0][0]:
                        if len(heap) == n_neighbors:
                            heapq.heappop(heap)
                        heapq.heappush(heap, (-dist, node['points'][i]))
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
                    dist_to_center = np.linalg.norm(point - node['center'])
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
