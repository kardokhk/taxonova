import logging
from typing import Dict, Set, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaxonomyHierarchy:
    """
    A class to store and enforce valid parent-child relationships in taxonomic hierarchy.
    
    This class maintains relationships between different taxonomic levels and provides
    methods to validate and correct inconsistent taxonomic classifications.
    """
    
    def __init__(self):
        """Initialize the taxonomy hierarchy structure."""
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        
        self.child_to_parent = {}
        
        self.parent_to_children = defaultdict(set)
        
        self.species_to_domain_paths = {}
        
        self.path_frequencies = defaultdict(int)
        
    def add_taxonomy(self, taxonomy: Dict[str, str]) -> None:
        """
        Add a taxonomy classification to the hierarchy.
        
        Args:
            taxonomy: A dictionary containing taxonomic assignments with levels as keys and taxa as values.
                     Now supports partial taxonomic lineages (domain and phylum required, others optional).
        """
        required_levels = ["domain", "phylum"]
        if not all(level in taxonomy and taxonomy[level] != "unknown" for level in required_levels):
            logger.warning(f"Skipping taxonomy without required levels {required_levels}: {taxonomy}")
            return
        
        available_levels = []
        for level in self.level_order:
            if level in taxonomy and taxonomy[level] != "unknown" and taxonomy[level].strip():
                available_levels.append(level)
        
        for i in range(len(available_levels) - 1, 0, -1):
            child_level = available_levels[i]
            parent_level = available_levels[i-1]
            
            child_value = taxonomy[child_level]
            parent_value = taxonomy[parent_level]
            
            child_key = (child_level, child_value)
            parent_key = (parent_level, parent_value)
            
            self.child_to_parent[child_key] = parent_key
            self.parent_to_children[parent_key].add(child_key)
        
        if len(available_levels) >= 2:
            path = []
            for level in self.level_order:
                if level in available_levels:
                    value = taxonomy[level]
                    path.append((level, value))
            
            path_tuple = tuple(path)
            self.path_frequencies[path_tuple] += 1
            
            most_specific_level = available_levels[-1]
            most_specific_value = taxonomy[most_specific_level]
            
            if most_specific_level == "species" and most_specific_value not in self.species_to_domain_paths:
                self.species_to_domain_paths[most_specific_value] = path
            elif most_specific_level == "species":
                existing_path = tuple(self.species_to_domain_paths[most_specific_value])
                if self.path_frequencies[path_tuple] > self.path_frequencies[existing_path]:
                    self.species_to_domain_paths[most_specific_value] = path
    
    def get_parent(self, level: str, value: str) -> Tuple[str, str]:
        """
        Get the parent taxon for a given taxon.
        
        Args:
            level: The taxonomic level of the taxon.
            value: The taxonomic value.
            
        Returns:
            A tuple of (parent_level, parent_value) if found, otherwise (None, None).
        """
        child_key = (level, value)
        if child_key in self.child_to_parent:
            return self.child_to_parent[child_key]
        return None, None
    
    def get_children(self, level: str, value: str) -> List[Tuple[str, str]]:
        """
        Get all valid children taxa for a given taxon.
        
        Args:
            level: The taxonomic level of the taxon.
            value: The taxonomic value.
            
        Returns:
            A list of (child_level, child_value) tuples.
        """
        parent_key = (level, value)
        return list(self.parent_to_children[parent_key])
    
    def get_consistent_path(self, taxonomy: Dict[str, str]) -> Dict[str, str]:
        """
        Enforce taxonomic consistency in a given taxonomy using a bottom-up approach.
        
        This method starts from the most specific available level and traces upward through the taxonomy,
        ensuring all parent-child relationships are consistent with the learned hierarchy.
        Now works with partial taxonomic lineages.
        
        Args:
            taxonomy: A dictionary containing taxonomic assignments with levels as keys and taxa as values.
            
        Returns:
            A dictionary containing the corrected taxonomy assignments.
        """
        consistent_taxonomy = taxonomy.copy()
        
        most_specific_level = None
        most_specific_value = None
        
        for i in range(len(self.level_order) - 1, -1, -1):
            current_level = self.level_order[i]
            current_value = taxonomy.get(current_level, "unknown")
            
            if current_value != "unknown" and current_value.strip():
                most_specific_level = current_level
                most_specific_value = current_value
                break
        
        if most_specific_level is None:
            return taxonomy
        
        if most_specific_level == "species" and most_specific_value in self.species_to_domain_paths:
            path = self.species_to_domain_paths[most_specific_value]
            for level, value in path:
                consistent_taxonomy[level] = value
            return consistent_taxonomy
        
        current_level = most_specific_level
        current_value = most_specific_value
        
        level_index = self.level_order.index(current_level)
        
        for i in range(level_index - 1, -1, -1):
            parent_level = self.level_order[i]
            parent_level_key, parent_value = self.get_parent(current_level, current_value)
            
            if parent_value is None:
                break
                
            consistent_taxonomy[parent_level] = parent_value
            
            current_level = parent_level
            current_value = parent_value
        
        return consistent_taxonomy
    
    def predict_parent_taxa(self, level: str, value: str) -> Dict[str, str]:
        """
        Predict the most likely parent taxa for a given taxon.
        
        Args:
            level: The taxonomic level of the taxon.
            value: The taxonomic value.
            
        Returns:
            A dictionary containing the predicted parent taxa.
        """
        result = {}
        
        if level == "species" and value in self.species_to_domain_paths:
            for level, value in self.species_to_domain_paths[value]:
                result[level] = value
            return result
        
        current_level = level
        current_value = value
        result[current_level] = current_value
        
        level_index = self.level_order.index(current_level)
        
        for i in range(level_index - 1, -1, -1):
            parent_level = self.level_order[i]
            parent_level_key, parent_value = self.get_parent(current_level, current_value)
            
            if parent_value is None:
                break
                
            result[parent_level] = parent_value
            current_level = parent_level
            current_value = parent_value
        
        return result
    
    def log_statistics(self) -> None:
        """Log statistics about the taxonomy hierarchy."""
        for i, level in enumerate(self.level_order):
            if i > 0:
                parent_level = self.level_order[i-1]
                child_count = sum(1 for key in self.child_to_parent if key[0] == level)
