from abc import ABC, abstractmethod

class Vectorizer(ABC):
    @abstractmethod
    def vectorize(self, data):
        pass
