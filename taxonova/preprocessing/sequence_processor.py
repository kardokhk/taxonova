from typing import Dict, List, Tuple, Union
import numpy as np
from collections import defaultdict

class SequenceProcessor:
    def __init__(self, k: Union[int, List[int]] = 8):
        if isinstance(k, int):
            self.k_values = [k]
        else:
            self.k_values = sorted(k)
            
        self.bases = ['A', 'C', 'G', 'T']
        
        self.features_per_k = {k_val: 4 ** k_val for k_val in self.k_values}
        self.num_features = sum(self.features_per_k.values())
        
        self._kmer_powers = {}
        for k_val in self.k_values:
            self._kmer_powers[k_val] = 4 ** np.arange(k_val - 1, -1, -1, dtype=np.int64)
        
        self._mapping = np.full(256, -1, dtype=np.int32)
        self._mapping[ord('A')] = 0
        self._mapping[ord('C')] = 1
        self._mapping[ord('G')] = 2
        self._mapping[ord('T')] = 3
        # RNA input: U is read as T, so RNA and DNA spellings of a read give the same result.
        self._mapping[ord('U')] = 3

    def generate_kmer_frequencies(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        seq_bytes = seq.encode('ascii', errors='ignore')
        
        seq_arr = np.frombuffer(seq_bytes, dtype=np.uint8)
        indices = self._mapping[seq_arr]
        
        combined_freqs = np.zeros(self.num_features, dtype=np.float32)
        
        feature_offset = 0
        for k_val in self.k_values:
            if len(indices) < k_val:
                feature_offset += self.features_per_k[k_val]
                continue
                
            shape = (len(indices) - k_val + 1, k_val)
            strides = (indices.strides[0], indices.strides[0])
            windows = np.lib.stride_tricks.as_strided(indices, shape=shape, strides=strides)
            
            valid_mask = np.all(windows >= 0, axis=1)
            valid_windows = windows[valid_mask]
            
            if valid_windows.size == 0:
                feature_offset += self.features_per_k[k_val]
                continue
                
            kmer_indices = valid_windows.dot(self._kmer_powers[k_val])
            counts = np.bincount(kmer_indices, minlength=self.features_per_k[k_val])
            
            freqs = counts.astype(np.float32)
            total = freqs.sum()
            if total > 0:
                freqs /= total
                
            combined_freqs[feature_offset:feature_offset + self.features_per_k[k_val]] = freqs
            feature_offset += self.features_per_k[k_val]
            
        return combined_freqs

    def _kmer_to_index(self, kmer: str) -> int:
        base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        index = 0
        for base in kmer:
            index = index * 4 + base_to_num[base]
        return index
