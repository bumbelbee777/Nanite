import re
import numpy as np
from gensim.models import KeyedVectors

class Word2VecTokenizer:
    """
    A standalone word-level tokenizer that:
      1. Loads a pretrained Word2Vec model (binary .bin or text .txt).
      2. Builds a vocab mapping word → index, with an <UNK> token.
      3. Provides tokenize/encode/decode methods.
      4. Exposes an embedding matrix you can plug into SillyAI’s input_proj.
    """
    def __init__(
        self,
        w2v_path: str,
        unk_token: str = "<UNK>",
        lowercase: bool = True
    ):
        # 1) Load pretrained word2vec
        #    (e.g. GoogleNews-vectors-negative300.bin)
        self.model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.lowercase = lowercase

        # 2) Build vocab + add UNK
        self.unk_token = unk_token
        self.index_to_word = list(self.model.index_to_key)[:]  # copy
        self.index_to_word.append(self.unk_token)
        self.word_to_index = {
            w: i for i, w in enumerate(self.index_to_word)
        }

        # 3) Build embedding matrix (numpy)
        D = self.model.vector_size
        # stack all vectors in order, then add a zero‐vector for UNK
        self.embedding_matrix = np.vstack([
            self.model[w] for w in self.model.index_to_key
        ] + [np.zeros(D, dtype=np.float32)])

    def tokenize(self, text: str) -> list[str]:
        """ Split on word boundaries, optionally lowercase. """
        if self.lowercase:
            text = text.lower()
        # simple regex tokenizer; you can swap for something else
        return re.findall(r"\b\w+\b", text)

    def encode(self, text: str) -> list[int]:
        """
        Turn raw text → list of token indices.
        OOV words map to self.unk_token.
        """
        tokens = self.tokenize(text)
        unk_idx = self.word_to_index[self.unk_token]
        return [self.word_to_index.get(tok, unk_idx) for tok in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        """ Map a list of indices back to words (UNK if out of range). """
        max_idx = len(self.index_to_word) - 1
        return [
            self.index_to_word[i] if 0 <= i <= max_idx else self.unk_token
            for i in indices
        ]

    def embed_indices(self, indices: list[int]) -> np.ndarray:
        """
        Fetch the embedding vectors for a list of indices.
        Returns: (len(indices), D) float32 array.
        """
        return self.embedding_matrix[indices]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Tokenize + encode + embed in one go.
        Returns: (num_tokens, D)
        """
        idxs = self.encode(text)
        return self.embed_indices(idxs)
