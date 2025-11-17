"""
Data loading utilities for Taxonova.
"""
import logging
from typing import List, Tuple

from taxonova.data.fasta import read_fasta_file

logger = logging.getLogger(__name__)

def load_sequences(fasta_file: str) -> List[Tuple]:
    """
    Load sequences from a FASTA file.
    
    Args:
        fasta_file: Path to the FASTA file
        
    Returns:
        List of sequence tuples
    """
    return read_fasta_file(fasta_file)
