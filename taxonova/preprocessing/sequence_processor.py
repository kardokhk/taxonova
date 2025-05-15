from typing import Dict, List, Tuple, Union
import numpy as np
from collections import defaultdict

class SequenceProcessor:
    def __init__(self, k: Union[int, List[int]] = 8):
        # Handle single k-mer length or multiple k-mer lengths
        if isinstance(k, int):
            self.k_values = [k]
        else:
            self.k_values = sorted(k)  # Sort k-values for consistent ordering
            
        self.bases = ['A', 'C', 'G', 'T']
        
        # Calculate features for each k-mer length and total features
        self.features_per_k = {k_val: 4 ** k_val for k_val in self.k_values}
        self.num_features = sum(self.features_per_k.values())
        
        # Precompute powers for each k-mer length
        self._kmer_powers = {}
        for k_val in self.k_values:
            self._kmer_powers[k_val] = 4 ** np.arange(k_val - 1, -1, -1, dtype=np.int64)
        
        # Precompute a mapping array for ASCII codes to base indices (A, C, G, T)
        self._mapping = np.full(256, -1, dtype=np.int32)
        self._mapping[ord('A')] = 0
        self._mapping[ord('C')] = 1
        self._mapping[ord('G')] = 2
        self._mapping[ord('T')] = 3

    def generate_kmer_frequencies(self, sequence: str) -> np.ndarray:
        # Ensure uppercase and encode to bytes
        seq = sequence.upper()
        seq_bytes = seq.encode('ascii', errors='ignore')
        
        # Convert to numpy array using the precomputed mapping
        seq_arr = np.frombuffer(seq_bytes, dtype=np.uint8)
        indices = self._mapping[seq_arr]
        
        # Initialize the combined feature vector
        combined_freqs = np.zeros(self.num_features, dtype=np.float32)
        
        # Process each k-mer length and combine frequencies
        feature_offset = 0
        for k_val in self.k_values:
            if len(indices) < k_val:
                # Skip this k-mer length if sequence is too short
                feature_offset += self.features_per_k[k_val]
                continue
                
            # Use stride tricks to create a 2D array of sliding windows
            shape = (len(indices) - k_val + 1, k_val)
            strides = (indices.strides[0], indices.strides[0])
            windows = np.lib.stride_tricks.as_strided(indices, shape=shape, strides=strides)
            
            # Filter out any windows with invalid bases (-1)
            valid_mask = np.all(windows >= 0, axis=1)
            valid_windows = windows[valid_mask]
            
            if valid_windows.size == 0:
                # Skip this k-mer length if no valid windows
                feature_offset += self.features_per_k[k_val]
                continue
                
            # Compute k-mer indices for valid windows
            kmer_indices = valid_windows.dot(self._kmer_powers[k_val])
            counts = np.bincount(kmer_indices, minlength=self.features_per_k[k_val])
            
            # Normalize frequencies for this k-mer length
            freqs = counts.astype(np.float32)
            total = freqs.sum()
            if total > 0:
                freqs /= total
                
            # Add to the combined feature vector
            combined_freqs[feature_offset:feature_offset + self.features_per_k[k_val]] = freqs
            feature_offset += self.features_per_k[k_val]
            
        return combined_freqs

    def _kmer_to_index(self, kmer: str) -> int:
        base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        index = 0
        for base in kmer:
            index = index * 4 + base_to_num[base]
        return index
