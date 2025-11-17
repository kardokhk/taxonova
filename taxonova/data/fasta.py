import re
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def parse_sequence_id(header: str) -> str:
    """Extract sequence ID from FASTA header, handling various formats."""
    match = re.match(r'^[>@]?(\S+)', header)
    if match:
        return match.group(1)
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
        full_header = header.lstrip('>')
        parts = full_header.split()
        if len(parts) < 2:
            if expect_taxonomy:
                logger.error(f"No taxonomy found in header: {header}")
            return taxonomy

        taxonomy_string = ' '.join(parts[1:])
        taxonomy_parts = [p.strip() for p in taxonomy_string.split(';')]
        
        first_part = taxonomy_parts[0]
        if 'd__' in first_part:
            domain = first_part.split('d__')[1].strip()
            taxonomy["domain"] = domain
        
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
                    if current_id and current_sequence:
                        sequence = ''.join(current_sequence)
                        taxonomy = parse_taxonomy(current_header, expect_taxonomy)
                        sequences.append((current_id, sequence, taxonomy))
                    
                    current_header = line
                    current_id = parse_sequence_id(line)
                    current_sequence = []
                else:
                    current_sequence.append(line)
            
            if current_id and current_sequence:
                sequence = ''.join(current_sequence)
                taxonomy = parse_taxonomy(current_header, expect_taxonomy)
                sequences.append((current_id, sequence, taxonomy))
                
        if len(sequences) > 0:
            if expect_taxonomy:
                first_tax = sequences[0][2]
                if first_tax["domain"] == "unknown":
                    logger.warning(f"First sequence domain parsed as unknown. Original header: {sequences[0][0]}")
        return sequences
    
    except Exception as e:
        logger.error(f"Error reading FASTA file {file_path}: {str(e)}")
        raise
