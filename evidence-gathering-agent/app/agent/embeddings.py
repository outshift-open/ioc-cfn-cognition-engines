import os
import yaml
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class EmbeddingManager:
    def __init__(self, config_path=None):
        if config_path is None:
            # Use path relative to this file's location
            config_path = os.path.join(os.path.dirname(__file__), "embeddings_config.yml")
        self.config = load_config(config_path)
        self.model_type = self.config.get("embedding_model_type", "huggingface")

        if self.model_type == "huggingface":
            self.model_name = self.config.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer(self.model_name)

        elif self.model_type == "openai":
            self.model_name = self.config.get("embedding_model_name", "")
            self.openai_key = self.config.get("openai_api_key", "")
            if not self.openai_key:
                print("Openai API key not found, falling back to the default huggingface model")
                self.model_type = "huggingface"
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)

        # fallback to HF embeddings model if no model is configured
        elif not self.model_type:
            self.model_type = "huggingface"
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)

    def preprocess_text(self, text, chunk_size=512, overlap=50):
        # Character length chunking
        # TODO : change method based on text type
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])

            start += chunk_size - overlap

        return chunks

    def generate_embeddings(self, text_chunks):
        embeddings = []
        if self.model_type == "huggingface":
            embeddings = self.model.encode(text_chunks, show_progress_bar=True)
        elif self.model_type == "openai":
            openai_client = OpenAI(self.openai_key)
            for text in text_chunks:
                response = openai_client.Embedding.create(input=text, model=self.model_name)
                embeddings.append(response["data"][0]["embedding"])
        return embeddings


if __name__ == "__main__":
    # Example test text
    test_text = "Collective intelligence: A trusted knowledge fabric for multi agent human societies"

    # Initialize EmbeddingManager with default config or specify a config path
    embedding_manager = EmbeddingManager()

    # Preprocess the text into chunks (optional, here just one chunk)
    text_chunks = embedding_manager.preprocess_text(test_text)

    # Generate embeddings for the chunks
    embeddings = embedding_manager.generate_embeddings(text_chunks)

    # Print the embeddings shape or content summary
    print(f"Generated {len(embeddings)} embeddings.")
    print(f"First embedding vector (truncated): {embeddings[0][:5]}")  # show first 5 values
