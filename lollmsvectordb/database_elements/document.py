from typing import Optional


class Document:
    """
    A class to represent a document in the database.

    Attributes:
    -----------
    id : Optional[int]
        The unique identifier for the document.
    hash : str
        The unique hash of the document.
    title : str
        The title of the document.
    path : str
        The file path of the document.
    """

    def __init__(self, hash: str, title: str, path: str, id: Optional[int] = None):
        """
        Constructs all the necessary attributes for the Document object.

        Parameters:
        -----------
        hash : str
            The unique hash of the document.
        title : str
            The title of the document.
        path : str
            The file path of the document.
        id : Optional[int]
            The unique identifier for the document. Default is None.
        """
        self.id = id
        self.hash = hash
        self.title = title
        self.path = path

    def __repr__(self) -> str:
        """
        Returns a string representation of the Document object.

        Returns:
        --------
        str
            A string representation of the Document object.
        """
        return f"Document(id={self.id}, hash={self.hash}, title={self.title}, path={self.path})"
