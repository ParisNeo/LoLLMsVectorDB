from abc import ABC, abstractmethod

class Vectorizer(ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
    @abstractmethod
    def vectorize(self, data):
        pass
