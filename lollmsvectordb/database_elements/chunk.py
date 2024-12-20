from typing import List, Optional

from lollmsvectordb.database_elements.document import Document


class Chunk:
    """
    A class to represent a chunk in the database.

    Attributes:
    -----------
    id : Optional[int]
        The unique identifier for the chunk.
    document_id : int
        The identifier of the document to which this chunk belongs.
    vector : bytes
        The vector representation of the chunk.
    text : str
        The text content of the chunk.
    nb_tokens : int
        The number of tokens in the chunk.
    """

    def __init__(
        self,
        doc: Document,
        vector: bytes,
        text: str,
        nb_tokens: int,
        id: Optional[int] = None,
        distance: float = 0,
        chunk_id: int = 0,
    ):
        """
        Constructs all the necessary attributes for the Chunk object.

        Parameters:
        -----------
        document_id : int
            The identifier of the document to which this chunk belongs.
        vector : bytes
            The vector representation of the chunk.
        text : str
            The text content of the chunk.
        nb_tokens : int
            The number of tokens in the chunk.
        id : Optional[int]
            The unique identifier for the chunk. Default is None.
        distance : Optional[float]
            The distance between the chunk and the query when searched. Default is None.
        """
        self.id = id
        self.chunk_id = chunk_id
        self.doc = doc
        self.vector = vector
        self.text = text
        self.nb_tokens = nb_tokens
        self.distance = distance

    def __repr__(self) -> str:
        """
        Returns a string representation of the Chunk object.

        Returns:
        --------
        str
            A string representation of the Chunk object.
        """
        return f"Chunk(id={self.id}, document_id={self.doc.id}, vector={self.vector}, text={self.text}, nb_tokens={self.nb_tokens}, chunk_id={self.chunk_id})"
