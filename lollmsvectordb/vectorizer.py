from abc import ABC, abstractmethod
from ascii_colors import ASCIIColors
class Vectorizer(ABC):
    """
    Abstract base class for vectorizers. A vectorizer is responsible for converting data into a vectorized form.
    
    Attributes:
        name (str): The name of the vectorizer.
        model (Any): The model used for vectorization. Initialized to None.
        requires_fitting (bool): Indicates whether the vectorizer requires fitting before vectorization.
    """
    
    def __init__(self, name: str, requires_fitting: bool = False) -> None:
        """
        Initializes the Vectorizer with a name and a flag indicating if fitting is required.
        
        Args:
            name (str): The name of the vectorizer.
            requires_fitting (bool): Whether the vectorizer requires fitting. Default is False.
        """
        super().__init__()
        self.name = name
        self.model_name = ""
        self.model = None
        self.parameters = None
        self.fitted = False
        self.requires_fitting = requires_fitting
        ASCIIColors.multicolor(["LollmsVectorDB>",f" Using vectorizer {name}"],[ASCIIColors.color_red, ASCIIColors.color_cyan])

    @abstractmethod
    def vectorize(self, data):
        """
        Abstract method to vectorize the given data. Must be implemented by subclasses.
        
        Args:
            data (Any): The data to be vectorized.
        
        Returns:
            Any: The vectorized form of the data.
        """
        pass

    def fit(self, data):
        """
        Fits the vectorizer to the given data. This method can be overridden by subclasses if fitting is required.
        
        Args:
            data (Any): The data to fit the vectorizer on.
        
        Returns:
            None
        """
        pass

    
    def get_models(self):
        """
        Returns a list of model names
        """
        return []