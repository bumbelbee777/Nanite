import re
import numpy as np
import torch
from gensim.models import KeyedVectors
from typing import List, Dict, Optional, Union
from pathlib import Path

class Word2VecTokenizer:
    """
    Enhanced word-level tokenizer with pretrained Word2Vec embeddings.
    
    Features:
    - Handles both binary (.bin) and text (.txt) Word2Vec formats
    - Customizable unknown token handling
    - Efficient batch processing
    - PyTorch integration
    - Subword fallback mechanism
    """
    
    def __init__(
        self,
        w2v_path: Union[str, Path],
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        lowercase: bool = True,
        subword_fallback: bool = False,
        device: str = "cpu"
    ):
        """
        Args:
            w2v_path: Path to Word2Vec model file
            unk_token: Token for unknown words
            pad_token: Token for padding
            lowercase: Convert text to lowercase
            subword_fallback: Try subword matching for OOV words
            device: Target device for PyTorch tensors
        """
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.lowercase = lowercase
        self.subword_fallback = subword_fallback
        self.device = device
        
        # Load Word2Vec model
        self._load_word2vec(w2v_path)
        self._build_vocab()
        self._create_embedding_matrix()
        
    def _load_word2vec(self, path: Union[str, Path]):
        """Handle different Word2Vec formats"""
        path = str(path)
        binary = path.endswith('.bin')
        
        try:
            self.model = KeyedVectors.load_word2vec_format(path, binary=binary)
            self.vector_size = self.model.vector_size
        except Exception as e:
            raise ValueError(f"Failed to load Word2Vec model from {path}: {str(e)}")

    def _build_vocab(self):
        """Build vocabulary with special tokens"""
        self.index_to_word = [self.pad_token, self.unk_token] + list(self.model.index_to_key)
        self.word_to_index = {w: i for i, w in enumerate(self.index_to_word)}
        
        # For subword fallback
        if self.subword_fallback:
            self._build_subword_index()

    def _build_subword_index(self):
        """Create n-gram index for subword fallback"""
        self.subword_index = {}
        for word in self.model.index_to_key:
            for i in range(len(word)):
                ngram = word[i:i+3]  # Use 3-grams
                if ngram not in self.subword_index:
                    self.subword_index[ngram] = []
                self.subword_index[ngram].append(word)

    def _create_embedding_matrix(self):
        """Initialize embedding matrix with special tokens"""
        # Pad token gets zeros, UNK gets random initialization
        pad_vec = np.zeros(self.vector_size, dtype=np.float32)
        unk_vec = np.random.normal(size=self.vector_size).astype(np.float32)
        
        # Stack all embeddings
        self.embedding_matrix = np.vstack([
            pad_vec,
            unk_vec,
            np.array([self.model[w] for w in self.model.index_to_key])
        ])
        
        # Convert to PyTorch tensor
        self.embedding_tensor = torch.from_numpy(self.embedding_matrix).to(self.device)

    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with optional subword fallback"""
        if self.lowercase:
            text = text.lower()
            
        # Improved regex pattern
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
        
        if self.subword_fallback:
            tokens = [self._handle_subword(tok) for tok in tokens]
            
        return tokens

    def _handle_subword(self, word: str) -> str:
        """Attempt subword matching for OOV words"""
        if word in self.word_to_index:
            return word
            
        # Try n-gram matching
        for i in range(len(word)):
            ngram = word[i:i+3]
            if ngram in self.subword_index:
                candidates = self.subword_index[ngram]
                # Return first match (could implement better scoring)
                return candidates[0]
                
        return self.unk_token

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: str = None
    ) -> Union[List[int], torch.Tensor]:
        """
        Enhanced encoding with padding options
        
        Args:
            text: Input text to encode
            max_length: Truncate/pad to this length
            padding: Whether to pad sequences
            return_tensors: 'pt' for PyTorch tensor, None for list
            
        Returns:
            List of indices or PyTorch tensor
        """
        tokens = self.tokenize(text)
        unk_idx = self.word_to_index[self.unk_token]
        indices = [self.word_to_index.get(tok, unk_idx) for tok in tokens]
        
        # Handle truncation/padding
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            elif padding:
                pad_idx = self.word_to_index[self.pad_token]
                indices += [pad_idx] * (max_length - len(indices))
                
        if return_tensors == 'pt':
            return torch.tensor(indices, device=self.device)
        return indices

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encoding for multiple texts
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoded = [self.encode(text, max_length, padding, 'pt') for text in texts]
        input_ids = torch.stack(encoded)
        
        # Create attention mask
        attention_mask = (input_ids != self.word_to_index[self.pad_token]).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def decode(self, indices: Union[List[int], torch.Tensor]) -> List[str]:
        """Decode indices back to text"""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        max_idx = len(self.index_to_word) - 1
        return [
            self.index_to_word[i] if 0 <= i <= max_idx else self.unk_token
            for i in indices
            if i != self.word_to_index[self.pad_token]  # Skip padding
        ]

    def embed(
        self,
        inputs: Union[str, List[int], torch.Tensor],
        return_tensors: str = 'pt'
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Flexible embedding lookup
        
        Args:
            inputs: Can be text, indices, or tensor
            return_tensors: 'pt' for PyTorch, 'np' for numpy
            
        Returns:
            Embeddings matrix (num_tokens, dim)
        """
        if isinstance(inputs, str):
            indices = self.encode(inputs, return_tensors='pt')
        elif isinstance(inputs, list):
            indices = torch.tensor(inputs, device=self.device)
        else:
            indices = inputs.to(self.device)
            
        embeddings = self.embedding_tensor[indices]
        
        if return_tensors == 'np':
            return embeddings.cpu().numpy()
        return embeddings

    def get_vocab_size(self) -> int:
        """Total vocabulary size including special tokens"""
        return len(self.index_to_word)

    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the full embedding matrix as PyTorch tensor"""
        return self.embedding_tensor

    def save_vocab(self, path: Union[str, Path]):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            for word in self.index_to_word:
                f.write(f"{word}\n")

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        vocab_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Create tokenizer from pretrained files
        
        Args:
            path: Path to Word2Vec model
            vocab_path: Optional custom vocabulary file
            **kwargs: Additional init arguments
        """
        tokenizer = cls(path, **kwargs)
        
        if vocab_path is not None:
            with open(vocab_path) as f:
                custom_vocab = [line.strip() for line in f]
            tokenizer.index_to_word = custom_vocab
            tokenizer.word_to_index = {w: i for i, w in enumerate(custom_vocab)}
            
        return tokenizer