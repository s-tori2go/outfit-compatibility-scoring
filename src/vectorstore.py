# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import sqlite3
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Literal
import numpy as np

from tqdm import tqdm
    
    
import os
import faiss
import pathlib
from collections import defaultdict

from . import vectorstore_utils


class FAISSVectorStore:
    def __init__(self, index_name: str = 'index', faiss_type: str = 'IndexFlatL2', 
                 base_dir: str = Path.cwd(), d_embed: int = 128, categories: Optional[List[str]] = None):
        self.base_dir = base_dir
        self.d_embed = d_embed
        self.faiss_type = faiss_type
        
        # General index
        index_path = os.path.join(base_dir, f"{index_name}.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = vectorstore_utils.create_faiss(faiss_type, d_embed)
        
        # Category-specific indices
        self.categories = categories
        self.category_indices = {}
        if categories:
            for cat in categories:
                cat_path = os.path.join(base_dir, f"{index_name}_{cat}.faiss")
                if os.path.exists(cat_path):
                    self.category_indices[cat] = faiss.read_index(cat_path)
                else:
                    self.category_indices[cat] = vectorstore_utils.create_faiss(faiss_type, d_embed)
    
    def add(self, embeddings: List[np.ndarray], ids: List[int], category: Optional[str] = None):
        if not embeddings or not ids:
            print(f"Skipping empty add for category {category}")
            return
            
        embeddings_np = np.array(embeddings).astype('float32')
        ids_np = np.array(ids, dtype=np.int64)
        
        # Ensure 2D shape: (n, d) even for single embedding
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        assert embeddings_np.ndim == 2, f"Embeddings must be 2D, got shape {embeddings_np.shape}"
        assert len(ids_np) == embeddings_np.shape[0], f"ID count {len(ids_np)} != embedding count {embeddings_np.shape[0]}"
        
        if category and category in self.category_indices:
            idx = self.category_indices[category]
        else:
            idx = self.index
        
        if hasattr(idx, 'add_with_ids'):
            idx.add_with_ids(embeddings_np, ids_np)
        else:
            idx.add(embeddings_np)

    def search(self, embeddings: List[np.ndarray], k: int = 10, category: Optional[str] = None):
        """Search embeddings. If category is specified, search only in that category index.
        Returns list of tuples (score, id)
        """
        if category and category in self.category_indices:
            idx = self.category_indices[category]
        else:
            idx = self.index
        
        embeddings_np = np.array(embeddings).astype('float32')
        scores, ids = idx.search(embeddings_np, k)
        results = []
        for s, i in zip(scores.tolist(), ids.tolist()):
            results.append(list(zip(s, i)))
        return results

    
    def save(self):
        # Save general index
        faiss.write_index(self.index, os.path.join(self.base_dir, "rec_index.faiss"))
        # Save category indices
        if self.categories:
            for cat, idx in self.category_indices.items():
                faiss.write_index(idx, os.path.join(self.base_dir, f"rec_index_{cat}.faiss"))
