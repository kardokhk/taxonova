import torch
import logging
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from taxonova.preprocessing.sequence_processor import SequenceProcessor

logger = logging.getLogger(__name__)

class TaxonomyDataset(Dataset):
    """Dataset for taxonomic sequence data using taxonomic classification."""
    
    def __init__(self, sequences: List[Tuple[str, str, Dict[str, str]]],
                 processor: SequenceProcessor,
                 label_encoder: Dict[str, int] = None,
                 domain_encoder: Dict[str, int] = None,
                 phylum_encoder: Dict[str, int] = None,
                 class_encoder: Dict[str, int] = None,
                 order_encoder: Dict[str, int] = None,
                 family_encoder: Dict[str, int] = None,
                 genus_encoder: Dict[str, int] = None,
                 species_encoder: Dict[str, int] = None):
        self.sequences = sequences
        self.processor = processor
        self.num_features = processor.num_features
        self.features = []
        self.sequence_ids = []
        
        # Taxonomic classification labels
        self.domain_labels = []
        self.phylum_labels = []
        self.class_labels = []
        self.order_labels = []
        self.family_labels = []
        self.genus_labels = []
        self.species_labels = []
        
        # Initialize encoders with provided values or empty dicts
        self.domain_encoder = domain_encoder or {}
        self.phylum_encoder = phylum_encoder or {}
        self.class_encoder = class_encoder or {}
        self.order_encoder = order_encoder or {}
        self.family_encoder = family_encoder or {}
        self.genus_encoder = genus_encoder or {}
        self.species_encoder = species_encoder or {}
        
        # Store individual levels for output formatting
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        
        # Track frequencies for reporting purposes
        self.taxa_frequencies = defaultdict(int)
        
        # For backward compatibility
        if label_encoder is not None:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = {}
        
        # Only create encoders if none are provided
        create_encoder = (not any([
            domain_encoder, phylum_encoder, class_encoder,
            order_encoder, family_encoder, genus_encoder,
            species_encoder
        ]))
        self._encode_sequences(create_encoder=create_encoder)
        
        self.features = np.array(self.features, dtype=np.float32)
        self.domain_labels = np.array(self.domain_labels) if self.domain_labels else np.array([])
        self.phylum_labels = np.array(self.phylum_labels) if self.phylum_labels else np.array([])
        self.class_labels = np.array(self.class_labels) if self.class_labels else np.array([])
        self.order_labels = np.array(self.order_labels) if self.order_labels else np.array([])
        self.family_labels = np.array(self.family_labels) if self.family_labels else np.array([])
        self.genus_labels = np.array(self.genus_labels) if self.genus_labels else np.array([])
        self.species_labels = np.array(self.species_labels) if self.species_labels else np.array([])
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx) -> Tuple:
        # Convert tensor index to integer if needed
        if torch.is_tensor(idx):
            idx = idx.item()
        elif isinstance(idx, np.ndarray):
            idx = idx.item()
            
        # Get features and convert to tensor
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # Return taxonomic labels for all levels
        return (feature, self.sequence_ids[idx]), (
            int(self.domain_labels[idx]),
            int(self.phylum_labels[idx]),
            int(self.class_labels[idx]),
            int(self.order_labels[idx]),
            int(self.family_labels[idx]),
            int(self.genus_labels[idx]),
            int(self.species_labels[idx])
        )

    def get_num_domain_classes(self) -> int:
        """Get number of domain classes."""
        return len(self.domain_encoder)
    
    def get_num_phylum_classes(self) -> int:
        """Get number of phylum classes."""
        return len(self.phylum_encoder)
    
    def get_num_class_classes(self) -> int:
        """Get number of class classes."""
        return len(self.class_encoder)
    
    def get_num_order_classes(self) -> int:
        """Get number of order classes."""
        return len(self.order_encoder)
    
    def get_num_family_classes(self) -> int:
        """Get number of family classes."""
        return len(self.family_encoder)
    
    def get_num_genus_classes(self) -> int:
        """Get number of genus classes."""
        return len(self.genus_encoder)
    
    def get_num_species_classes(self) -> int:
        """Get number of species classes."""
        return len(self.species_encoder)
    
    def _track_taxa_frequencies(self):
        """Track frequency of each taxonomic group."""
        for _, _, taxonomy in self.sequences:
            for level in self.level_order:
                taxon = taxonomy.get(level, "unknown")
                self.taxa_frequencies[f"{level}_{taxon}"] += 1

    def get_rare_taxa(self, threshold: int = 1) -> Dict[str, List[str]]:
        """Get taxa that appear less than or equal to threshold times by level."""
        rare_taxa = {}
        for level in self.level_order:
            rare_taxa[level] = []
            for taxa_key, count in self.taxa_frequencies.items():
                if count <= threshold and taxa_key.startswith(f"{level}_") and not taxa_key.endswith("unknown"):
                    rare_taxa[level].append(taxa_key.replace(f"{level}_", ""))
        return rare_taxa

    def _encode_sequences(self, create_encoder: bool = True):
        """Encode sequences with taxonomy labels."""
        # Track taxa frequencies first
        self._track_taxa_frequencies()

        if create_encoder:
            # Initialize sets for all taxonomic levels
            all_domains = set()
            all_phyla = set()
            all_classes = set()
            all_orders = set()
            all_families = set()
            all_genera = set()
            all_species = set()
            
            # First pass to collect all unique taxonomic values
            for _, _, taxonomy in self.sequences:
                domain = taxonomy.get("domain", "unknown")
                phylum = taxonomy.get("phylum", "unknown")
                class_name = taxonomy.get("class", "unknown")
                order = taxonomy.get("order", "unknown")
                family = taxonomy.get("family", "unknown")
                genus = taxonomy.get("genus", "unknown")
                species = taxonomy.get("species", "unknown")
                
                all_domains.add(domain)
                all_phyla.add(phylum)
                all_classes.add(class_name)
                all_orders.add(order)
                all_families.add(family)
                all_genera.add(genus)
                all_species.add(species)
            
            # Create sorted lists for deterministic encoding
            sorted_domains = sorted(list(all_domains))
            sorted_phyla = sorted(list(all_phyla))
            sorted_classes = sorted(list(all_classes))
            sorted_orders = sorted(list(all_orders))
            sorted_families = sorted(list(all_families))
            sorted_genera = sorted(list(all_genera))
            sorted_species = sorted(list(all_species))
            
            # Ensure "unknown" is always first (index 0)
            for sorted_list in [sorted_domains, sorted_phyla, sorted_classes, sorted_orders, 
                              sorted_families, sorted_genera, sorted_species]:
                if "unknown" in sorted_list:
                    sorted_list.remove("unknown")
                sorted_list.insert(0, "unknown")
            
            # Create encoders for all levels
            self.domain_encoder = {domain: idx for idx, domain in enumerate(sorted_domains)}
            self.phylum_encoder = {phylum: idx for idx, phylum in enumerate(sorted_phyla)}
            self.class_encoder = {class_name: idx for idx, class_name in enumerate(sorted_classes)}
            self.order_encoder = {order: idx for idx, order in enumerate(sorted_orders)}
            self.family_encoder = {family: idx for idx, family in enumerate(sorted_families)}
            self.genus_encoder = {genus: idx for idx, genus in enumerate(sorted_genera)}
            self.species_encoder = {species: idx for idx, species in enumerate(sorted_species)}

        # Encode sequences
        for seq_id, sequence, taxonomy in self.sequences:
            # Store sequence identifier
            self.sequence_ids.append(seq_id)
            
            # Process sequence features
            feats = self.processor.generate_kmer_frequencies(sequence)
            self.features.append(feats)
            
            # Extract all taxonomic levels
            domain = taxonomy.get("domain", "unknown")
            phylum = taxonomy.get("phylum", "unknown")
            class_name = taxonomy.get("class", "unknown")
            order = taxonomy.get("order", "unknown")
            family = taxonomy.get("family", "unknown")
            genus = taxonomy.get("genus", "unknown")
            species = taxonomy.get("species", "unknown")
            
            # Encode all levels with default to "unknown" (0)
            domain_idx = self.domain_encoder.get(domain, 0)
            phylum_idx = self.phylum_encoder.get(phylum, 0)
            class_idx = self.class_encoder.get(class_name, 0)
            order_idx = self.order_encoder.get(order, 0)
            family_idx = self.family_encoder.get(family, 0)
            genus_idx = self.genus_encoder.get(genus, 0)
            species_idx = self.species_encoder.get(species, 0)
            
            # Store all encoded labels
            self.domain_labels.append(domain_idx)
            self.phylum_labels.append(phylum_idx)
            self.class_labels.append(class_idx)
            self.order_labels.append(order_idx)
            self.family_labels.append(family_idx)
            self.genus_labels.append(genus_idx)
            self.species_labels.append(species_idx)
    
    def get_domain_label(self, domain_idx: int) -> str:
        """Convert domain index back to domain name."""
        for domain, idx in self.domain_encoder.items():
            if idx == domain_idx:
                return domain
        return "unknown"
    
    def get_phylum_label(self, phylum_idx: int) -> str:
        """Convert phylum index back to phylum name."""
        for phylum, idx in self.phylum_encoder.items():
            if idx == phylum_idx:
                return phylum
        return "unknown"
    
    def get_class_label(self, class_idx: int) -> str:
        """Convert class index back to class name."""
        for class_name, idx in self.class_encoder.items():
            if idx == class_idx:
                return class_name
        return "unknown"
    
    def get_order_label(self, order_idx: int) -> str:
        """Convert order index back to order name."""
        for order, idx in self.order_encoder.items():
            if idx == order_idx:
                return order
        return "unknown"
    
    def get_family_label(self, family_idx: int) -> str:
        """Convert family index back to family name."""
        for family, idx in self.family_encoder.items():
            if idx == family_idx:
                return family
        return "unknown"
    
    def get_genus_label(self, genus_idx: int) -> str:
        """Convert genus index back to genus name."""
        for genus, idx in self.genus_encoder.items():
            if idx == genus_idx:
                return genus
        return "unknown"
    
    def get_species_label(self, species_idx: int) -> str:
        """Convert species index back to species name."""
        for species, idx in self.species_encoder.items():
            if idx == species_idx:
                return species
        return "unknown"
