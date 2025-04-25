# LoLLMsVectorDB

**LoLLMsVectorDB**: A modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem. Supports various vectorization methods and directory bindings for efficient text data management.

## Features

- **Flexible Vectorization**: Supports multiple vectorization methods including TF-IDF and Word2Vec.
- **Directory Binding**: Automatically updates the vector store with text data from a specified directory.
- **Efficient Search**: Provides fast and accurate search results with metadata to locate the original text chunks.
- **Modular Design**: Easily extendable to support new vectorization methods and functionalities.

## Installation

```bash
pip install lollmsvectordb
```

## Usage

### Example with TFIDFVectorizer

```python
from lollmsvectordb import TFIDFVectorizer, VectorDatabase, DirectoryBinding

# Initialize the vectorizer
tfidf_vectorizer = TFIDFVectorizer()
tfidf_vectorizer.fit(["This is a sample text.", "Another sample text."])

# Create the vector database
db = VectorDatabase("vector_db.sqlite", tfidf_vectorizer)

# Bind a directory to the vector database
directory_binding = DirectoryBinding("path_to_your_directory", db)

# Update the vector store with text data from the directory
directory_binding.update_vector_store()

# Search for a query in the vector database
results = directory_binding.search("This is a sample text.")
print(results)
```

### Example with OllamaVectorizer and bge-m3 embedding model
```python
from lollmsvectordb import VectorDatabase, DirectoryBinding
from lollmsvectordb.lollms_vectorizers.ollama_vectorizer import OllamaVectorizer

doc_path = "/path/to/your/folder"

# Initialize the vectorizer
ollama_vectorizer = OllamaVectorizer(model_name="bge-m3", url="http://my_ollama_server:11434")

# Create the vector database
db = VectorDatabase("vector_db.sqlite", ollama_vectorizer)

# Bind a directory to the vector database
directory_binding = DirectoryBinding(doc_path, db)

# Update the vector store with text data from the directory
directory_binding.update_vector_store()  # Comment this line for retrieval only.

# Search for a query in the vector database
results = directory_binding.search("What is the database you have access to about?")
print(results)
```

### Adding New Vectorization Methods

To add a new vectorization method, create a subclass of the `Vectorizer` class and implement the `vectorize` method.

```python
from lollmsvectordb import Vectorizer

class CustomVectorizer(Vectorizer):
    def vectorize(self, data):
        # Implement your custom vectorization logic here
        pass
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, feel free to reach out to the author:

- **Twitter**: [@ParisNeo_AI](https://twitter.com/ParisNeo_AI)
- **Discord**: [Join our Discord](https://discord.gg/BDxacQmv)
- **Sub-Reddit**: [r/lollms](https://www.reddit.com/r/lollms/)
- **Instagram**: [spacenerduino](https://www.instagram.com/spacenerduino/)

## Acknowledgements

Special thanks to the LoLLMs community for their continuous support and contributions.
