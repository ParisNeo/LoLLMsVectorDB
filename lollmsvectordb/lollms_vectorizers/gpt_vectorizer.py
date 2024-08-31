from transformers import GPT2Tokenizer, GPT2Model
from lollmsvectordb.vectorizer import Vectorizer

class GPTVectorizer(Vectorizer):
    def __init__(self, model_name: str = 'gpt2'):
        super().__init__("GPTVectorizer")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model_ = GPT2Model.from_pretrained(model_name)

    def vectorize(self, data: List[str]) -> List[np.ndarray]:
        embeddings = []
        for text in data:
            inputs = self.tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model_(**inputs)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(sentence_embedding)
        return embeddings

    def __str__(self):
        return 'Lollms Vector DB GPTVectorizer.'

    def __repr__(self):
        return 'Lollms Vector DB GPTVectorizer.'
