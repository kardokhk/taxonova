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
        # Define the taxonomic levels in hierarchical order (from high to low)
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        
        # Store child-parent relationships for each taxonomic level
        # {(child_level, child_value): (parent_level, parent_value)}
        self.child_to_parent = {}
        
        # Store parent-child relationships for each taxonomic level
        # {(parent_level, parent_value): set((child_level, child_values))}
        self.parent_to_children = defaultdict(set)
        
        # Full paths from species to domain for each species
        # {species_value: [(level, value), ...]}
        self.species_to_domain_paths = {}
        
        # Track the frequency of each path to handle conflicts
        self.path_frequencies = defaultdict(int)
        
    def add_taxonomy(self, taxonomy: Dict[str, str]) -> None:
        """
        Add a taxonomy classification to the hierarchy.
        
        Args:
            taxonomy: A dictionary containing taxonomic assignments with levels as keys and taxa as values.
        """
        # Skip incomplete taxonomies
        if not all(level in taxonomy for level in self.level_order):
            return
        
        # Build child-parent relationships for consecutive levels
        for i in range(len(self.level_order) - 1, 0, -1):
            child_level = self.level_order[i]
            parent_level = self.level_order[i-1]
            
            child_value = taxonomy[child_level]
            parent_value = taxonomy[parent_level]
            
            if child_value == "unknown" or parent_value == "unknown":
                continue
                
            # Store the child-parent relationship
            child_key = (child_level, child_value)
            parent_key = (parent_level, parent_value)
            
            self.child_to_parent[child_key] = parent_key
            self.parent_to_children[parent_key].add(child_key)
        
        # Store the full path for this species
        species_value = taxonomy["species"]
        if species_value != "unknown":
            path = []
            for level in self.level_order:
                value = taxonomy[level]
                if value != "unknown":
                    path.append((level, value))
            
            if len(path) == len(self.level_order):
                path_tuple = tuple(path)
                self.path_frequencies[path_tuple] += 1
                
                # Store or update the species path if it's more frequent
                if species_value not in self.species_to_domain_paths or \
                   self.path_frequencies[path_tuple] > self.path_frequencies[tuple(self.species_to_domain_paths[species_value])]:
                    self.species_to_domain_paths[species_value] = path
    
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
        
        This method starts from the species level and traces upward through the taxonomy,
        ensuring all parent-child relationships are consistent with the learned hierarchy.
        
        Args:
            taxonomy: A dictionary containing taxonomic assignments with levels as keys and taxa as values.
            
        Returns:
            A dictionary containing the corrected taxonomy assignments.
        """
        # Copy the input taxonomy to avoid modifying it
        consistent_taxonomy = taxonomy.copy()
        
        # Start with the most specific level (species) that is not "unknown"
        for i in range(len(self.level_order) - 1, -1, -1):
            current_level = self.level_order[i]
            current_value = taxonomy.get(current_level, "unknown")
            
            if current_value != "unknown":
                # If we have a valid species, use its full path
                if current_level == "species" and current_value in self.species_to_domain_paths:
                    path = self.species_to_domain_paths[current_value]
                    for level, value in path:
                        consistent_taxonomy[level] = value
                    return consistent_taxonomy
                
                # Otherwise, start from the current level and trace upwards
                break
        else:
            # No valid taxa found, return the original taxonomy
            return taxonomy
        
        # Start with the most specific valid level and trace upwards
        current_level = self.level_order[i]
        current_value = taxonomy.get(current_level, "unknown")
        
        # Trace upward through the taxonomic hierarchy
        while i > 0:
            parent_level, parent_value = self.get_parent(current_level, current_value)
            
            if parent_level is None:
                # No parent found, keep the original value
                break
                
            # Set the parent value in the consistent taxonomy
            consistent_taxonomy[parent_level] = parent_value
            
            # Move up to the parent
            current_level = parent_level
            current_value = parent_value
            i -= 1
        
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
        
        # If it's a species, use the stored path if available
        if level == "species" and value in self.species_to_domain_paths:
            for level, value in self.species_to_domain_paths[value]:
                result[level] = value
            return result
        
        # Otherwise, trace upward through the hierarchy
        current_level = level
        current_value = value
        result[current_level] = current_value
        
        level_index = self.level_order.index(current_level)
        
        # Trace upward
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
        logger.info(f"Taxonomy hierarchy statistics:")
        for i, level in enumerate(self.level_order):
            if i > 0:
                parent_level = self.level_order[i-1]
                child_count = sum(1 for key in self.child_to_parent if key[0] == level)
                logger.info(f"  {level}->{parent_level} relationships: {child_count}")
        
        logger.info(f"Species with complete taxonomic paths: {len(self.species_to_domain_paths)}")
