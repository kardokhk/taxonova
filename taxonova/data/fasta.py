import re
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def parse_sequence_id(header: str) -> str:
    """Extract sequence ID from FASTA header, handling various formats."""
    # Try to get first word before any spaces or special characters
    match = re.match(r'^[>@]?(\S+)', header)
    if match:
        return match.group(1)
    # Fallback to the full header without > character
    return header.lstrip('>@').strip()

def parse_taxonomy(header: str, expect_taxonomy: bool = True) -> Dict[str, str]:
    """Parse taxonomy information from FASTA header.
    
    Expected format:
    >sequence_id d__Bacteria; p__Phylum; c__Class; o__Order; f__Family; g__Genus; s__Species name
    """
    taxonomy = {
        "domain": "unknown",
        "phylum": "unknown",
        "class": "unknown",
        "order": "unknown",
        "family": "unknown",
        "genus": "unknown",
        "species": "unknown"
    }
    
    try:
        # Get taxonomy part after sequence ID
        full_header = header.lstrip('>')
        # Handle multiple whitespace characters between ID and taxonomy
        parts = full_header.split()
        if len(parts) < 2:
            if expect_taxonomy:
                logger.error(f"No taxonomy found in header: {header}")
            return taxonomy

        # Get the taxonomy string and split into parts
        taxonomy_string = ' '.join(parts[1:])  # Everything after ID
        taxonomy_parts = [p.strip() for p in taxonomy_string.split(';')]
        
        # First part contains domain - extract it
        first_part = taxonomy_parts[0]
        if 'd__' in first_part:
            domain = first_part.split('d__')[1].strip()
            taxonomy["domain"] = domain
        
        # Process remaining parts
        for part in taxonomy_parts:
            part = part.strip()
            if '__' not in part:
                continue
                
            prefix, value = part.split('__', 1)
            prefix = prefix.strip().lower()
            value = value.strip()
            
            if prefix == 'd':
                taxonomy["domain"] = value
            elif prefix == 'p':
                taxonomy["phylum"] = value
            elif prefix == 'c':
                taxonomy["class"] = value
            elif prefix == 'o':
                taxonomy["order"] = value
            elif prefix == 'f':
                taxonomy["family"] = value
            elif prefix == 'g':
                taxonomy["genus"] = value
            elif prefix == 's':
                taxonomy["species"] = value
                
    except Exception as e:
        logger.warning(f"Error parsing taxonomy from header: {header}. Error: {str(e)}")
        
    return taxonomy

def read_fasta_file(file_path: str, expect_taxonomy: bool = True) -> List[Tuple[str, str, Dict[str, str]]]:
    """Read sequences and their taxonomic information from a FASTA file.
    
    Args:
        file_path: Path to the FASTA file
        expect_taxonomy: Whether to expect taxonomy information in headers (default: True)
        
    Returns:
        List of tuples (sequence_id, sequence, taxonomy_dict)
    """
    sequences = []
    current_sequence = []
    current_id = None
    current_header = None
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_id and current_sequence:
                        sequence = ''.join(current_sequence)
                        taxonomy = parse_taxonomy(current_header, expect_taxonomy)
                        sequences.append((current_id, sequence, taxonomy))
                    
                    # Start new sequence
                    current_header = line  # Keep complete header including '>'
                    current_id = parse_sequence_id(line)
                    current_sequence = []
                else:
                    current_sequence.append(line)
            
            # Add last sequence
            if current_id and current_sequence:
                sequence = ''.join(current_sequence)
                taxonomy = parse_taxonomy(current_header, expect_taxonomy)
                sequences.append((current_id, sequence, taxonomy))
                
        # Log examples and check for unknowns
        if len(sequences) > 0:
            logger.info(f"Read {len(sequences)} sequences from {file_path}")
            if expect_taxonomy:
                first_tax = sequences[0][2]
                logger.info(f"Example taxonomy from first sequence: {first_tax}")
                if first_tax["domain"] == "unknown":
                    logger.warning(f"First sequence domain parsed as unknown. Original header: {sequences[0][0]}")
        return sequences
    
    except Exception as e:
        logger.error(f"Error reading FASTA file {file_path}: {str(e)}")
        raise
