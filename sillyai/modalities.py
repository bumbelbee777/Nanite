import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
from .config import ModelConfig
from gensim.models import KeyedVectors
from .core import LinearLayer
import re

from .ops import MultivectorOps, tensor_to_multivector

class ImageModality(nn.Module):
    def __init__(
        self,
        ops: MultivectorOps,
        input_channels: int = 3,
        output_dim: int = 256,
        channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: Optional[List[int]] = None,
        pool_type: str = "adaptive",            # "adaptive" | "max" | "none"
        pool_size: Tuple[int,int] = (4,4),       # for adaptive
        dropout: float = 0.3,
        activation: str = "gelu",
        use_batchnorm: bool = True,
        min_resolution: int = 32
    ):
        super().__init__()
        assert pool_type in ("adaptive","max","none")
        
        self.ops            = ops
        self.input_channels = input_channels
        self.output_dim     = output_dim
        self.channels       = channels
        self.kernel_sizes   = kernel_sizes or [3]*len(channels)
        self.pool_type      = pool_type
        self.pool_size      = pool_size
        self.use_bn         = use_batchnorm
        self.min_resolution = min_resolution
        
        # Build conv params for each block
        self.convs = nn.ModuleList()
        in_c = input_channels
        for out_c, k in zip(channels, self.kernel_sizes):
            conv = nn.ModuleDict({
                "weight": nn.Parameter(torch.randn(out_c, in_c, k, k, dtype=torch.complex64)*0.02),
                "bias":   nn.Parameter(torch.zeros(out_c, dtype=torch.complex64)),
                # real & imag BatchNorm
                "bn_real": nn.BatchNorm2d(out_c) if use_batchnorm else None,
                "bn_imag": nn.BatchNorm2d(out_c) if use_batchnorm else None,
                "act":     getattr(nn, activation.title())() if hasattr(nn, activation.title()) else nn.GELU(),
            })
            self.convs.append(conv)
            in_c = out_c
        
        # Pooling
        if pool_type == "adaptive":
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = None
        
        # FC as weight & bias (complex)
        # flatten_dim computed lazily on first forward if pool_type=="none"
        self.fc_weight = nn.Parameter(torch.randn(1, output_dim, dtype=torch.complex64))  # placeholder
        self.fc_bias   = nn.Parameter(torch.zeros(output_dim, dtype=torch.complex64))
        self._flatten_dim = None
        
        self.dropout = nn.Dropout2d(dropout)
    
    async def forward(self,
                      x: torch.Tensor,
                      input_size: Optional[Tuple[int,int]] = None
                     ) -> torch.Tensor:
        """
        x: real tensor [B, C, H, W]
        returns: complex multivector [B, output_dim, mv_dim]
        """
        B, C, H, W = x.shape
        assert C == self.input_channels
        assert H >= self.min_resolution and W >= self.min_resolution
        
        # promote to complex
        x_c = x.to(torch.complex64)
        
        # conv blocks
        for conv in self.convs:
            w, b = conv["weight"], conv["bias"]
            x_c = await self.ops.conv2d(x_c, w, b)      # complex conv
            if self.use_bn:
                real = conv["bn_real"](x_c.real)
                imag = conv["bn_imag"](x_c.imag)
                x_c = torch.complex(real, imag)
            x_c = conv["act"](x_c)
            x_c = self.dropout(x_c)
        
        # pooling / flatten
        if self.pool:
            # apply to real & imag separately
            r = self.pool(x_c.real)
            i = self.pool(x_c.imag)
            x_flat = torch.complex(r, i).flatten(1)
        else:
            # no pooling: flatten full H*W → determine dim once
            flat = x_c.flatten(1)  # [B, C*H*W]
            if self._flatten_dim is None:
                self._flatten_dim = flat.shape[-1]
                # reinit fc weight to correct shape
                self.fc_weight.data = torch.randn(self._flatten_dim, self.output_dim, dtype=torch.complex64)
            x_flat = flat
        
        # linear projection
        # [B, flatten_dim] @ [flatten_dim, output_dim] => [B, output_dim]
        out = await self.ops.matmul(x_flat, self.fc_weight)
        out = out + self.fc_bias
        
        # embed as multivector for SillyAI
        mv = tensor_to_multivector(out, n_dims=None)
        return mv

    def forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        """Sync path for ONNX export; mirrors async logic."""
        B, C, H, W = x.shape
        x_c = x.to(torch.complex64)
        for conv in self.convs:
            w, b = conv["weight"], conv["bias"]
            x_c = self.ops.conv2d(x_c, w, b)  
            if self.use_bn:
                real = conv["bn_real"](x_c.real)
                imag = conv["bn_imag"](x_c.imag)
                x_c = torch.complex(real, imag)
            x_c = conv["act"](x_c)
            x_c = self.dropout(x_c)
        
        if self.pool:
            r = self.pool(x_c.real);  i = self.pool(x_c.imag)
            x_flat = torch.complex(r,i).flatten(1)
        else:
            flat = x_c.flatten(1)
            if self._flatten_dim is None:
                self._flatten_dim = flat.shape[-1]
                self.fc_weight.data = torch.randn(self._flatten_dim, self.output_dim, dtype=torch.complex64)
            x_flat = flat
        
        out = self.ops.matmul_sync(x_flat, self.fc_weight) + self.fc_bias
        mv  = tensor_to_multivector(out, n_dims=None)
        return mv

class Word2VecTokenizer:
    """
    Word2Vec tokenizer that produces complex-valued multivector embeddings
    ready for SillyAI.
    """
    def __init__(
        self,
        w2v_path: Union[str, Path],
        ops: MultivectorOps,
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        lowercase: bool = True,
        subword_fallback: bool = False,
        device: str = "cpu",
        imag_scale: float = 0.0
    ):
        """
        Args:
            w2v_path: Path to .bin or .txt Word2Vec file.
            ops:      MultivectorOps instance for later pipeline.
            imag_scale: stddev for random imaginary part (0=zero imag).
        """
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.lowercase = lowercase
        self.subword_fallback = subword_fallback
        self.device = device
        self.ops = ops
        self.imag_scale = imag_scale

        # load and build vocab
        self._load_w2v(w2v_path)
        self._build_vocab()
        self._make_embedding()

    def _load_w2v(self, path: Union[str, Path]):
        path = str(path)
        binary = path.endswith(".bin")
        self.model = KeyedVectors.load_word2vec_format(path, binary=binary)
        self.vec_size = self.model.vector_size

    def _build_vocab(self):
        keys = list(self.model.index_to_key)
        self.index_to_word = [self.pad_token, self.unk_token] + keys
        self.word_to_index = {w: i for i, w in enumerate(self.index_to_word)}
        if self.subword_fallback:
            self._build_subword_index(keys)

    def _build_subword_index(self, keys: List[str]):
        self.subword_index: Dict[str, List[str]] = {}
        for w in keys:
            for i in range(len(w)-2):
                tri = w[i:i+3]
                self.subword_index.setdefault(tri, []).append(w)

    def _make_embedding(self):
        # real part from W2V, imag part random (or zero)
        pad_vec = np.zeros(self.vec_size, np.float32)
        unk_vec = np.random.normal(size=self.vec_size).astype(np.float32)
        real_mat = np.vstack([
            pad_vec,
            unk_vec,
            np.array([self.model[w] for w in self.model.index_to_key], np.float32)
        ])
        imag_mat = np.random.normal(
            scale=self.imag_scale,
            size=real_mat.shape
        ).astype(np.float32) if self.imag_scale > 0 else np.zeros_like(real_mat)
        complex_mat = real_mat + 1j * imag_mat

        # convert to torch complex tensor
        self.embedding = torch.from_numpy(complex_mat).to(self.device)

    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        toks = re.findall(r"\b\w+(?:'\w+)?\b", text)
        if self.subword_fallback:
            return [self._subword(tok) for tok in toks]
        return toks

    def _subword(self, tok: str) -> str:
        if tok in self.word_to_index:
            return tok
        for i in range(len(tok)-2):
            tri = tok[i:i+3]
            if tri in self.subword_index:
                return self.subword_index[tri][0]
        return self.unk_token

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        toks = self.tokenize(text)
        unk_idx = self.word_to_index[self.unk_token]
        ids = [self.word_to_index.get(t, unk_idx) for t in toks]
        if max_length:
            pad_idx = self.word_to_index[self.pad_token]
            if len(ids) > max_length:
                ids = ids[:max_length]
            elif padding:
                ids += [pad_idx] * (max_length - len(ids))
        if return_tensors == 'pt':
            return torch.tensor(ids, device=self.device)
        return ids

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        encs = [self.encode(t, max_length, padding=True, return_tensors='pt')
                for t in texts]
        input_ids = torch.stack(encs, dim=0)            # [B, S]
        attention_mask = (input_ids != self.word_to_index[self.pad_token])\
                         .to(torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def decode(self, ids: Union[List[int], torch.Tensor]) -> List[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        max_i = len(self.index_to_word)-1
        return [self.index_to_word[i] if 0 <= i <= max_i else self.unk_token
                for i in ids if i != self.word_to_index[self.pad_token]]

    async def embed_async(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Async lookup → complex embeddings → multivector channels.
        input_ids: [B, S] long tensor
        returns:  [B, S, vec_size, mv_dim]
        """
        # [B, S, D]
        emb = self.embedding[input_ids]
        # reshape to [B*S, D]
        B, S, D = emb.shape
        flat = emb.view(B*S, D)
        # optionally apply an MV Op, e.g. a learned linear projection
        # here we skip, and directly convert
        complex_feats = flat                                # already complex
        mv = tensor_to_multivector(
            complex_feats.view(B, S, D),
            n_dims=None
        )
        return mv

    def embed(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Sync version for ONNX or direct use.
        """
        emb = self.embedding[input_ids]
        B, S, D = emb.shape
        mv = tensor_to_multivector(emb, n_dims=None)
        return mv

    def get_vocab_size(self) -> int:
        return len(self.index_to_word)

    def get_embedding_matrix(self) -> torch.Tensor:
        return self.embedding

    @classmethod
    def from_pretrained(
        cls,
        w2v_path: Union[str, Path],
        ops: MultivectorOps,
        **kwargs: Any
    ):
        return cls(w2v_path, ops, **kwargs)

class ModalityManager(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.modalities = nn.ModuleDict()
        
    def add_modality(self, name: str, modality_config: dict):
        if name == 'text':
            self._init_text_modality(modality_config)
        elif name == 'image':
            self._init_image_modality(modality_config)
            
    def _init_text_modality(self, config):
        self.modalities['text_tokenizer'] = Word2VecTokenizer(
            w2v_path=config['w2v_path'],
            unk_token=config.get('unk_token', '<UNK>'),
            pad_token=config.get('pad_token', '<PAD>'),
            lowercase=config.get('lowercase', True),
            device=self.config.device
        )
        self.modalities['text_proj'] = LinearLayer(
            self.modalities['text_tokenizer'].get_embedding_matrix().shape[1],
            self.config.d_model,
            factorized=self.config.factorized_linear
        )
        
    def _init_image_modality(self, config):
        self.modalities['image_encoder'] = ImageModality(
            input_channels=config.get('input_channels', 3),
            output_dim=self.config.d_model,
            input_size=config.get('input_size', (224, 224)),
            dropout_rate=config.get('dropout_rate', 0.1)
        )
        self.modalities['image_proj'] = LinearLayer(
            self.config.d_model,
            self.config.d_model,
            factorized=self.config.factorized_linear
        )
        
    def process_text(self, text):
        if isinstance(text, str):
            text = [text]
        encoded = self.modalities['text_tokenizer'].batch_encode(text)
        text_emb = self.modalities['text_tokenizer'].embed(encoded['input_ids'])
        return self.modalities['text_proj'](text_emb)
        
    def process_image(self, images):
        img_features = self.modalities['image_encoder'](images)
        return self.modalities['image_proj'](img_features)