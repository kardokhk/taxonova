import torch
import logging
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.utils.taxonomy_hierarchy import TaxonomyHierarchy

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
                 species_encoder: Dict[str, int] = None,
                 taxonomy_hierarchy: TaxonomyHierarchy = None,
                 length_aug_p: float = 0.0,
                 length_aug_min: int = 100,
                 length_aug_max: int = 10**9):
        self.sequences = sequences
        self.processor = processor
        self.num_features = processor.num_features
        # Pre-allocate the feature cache as a single contiguous ndarray.
        # On V1-V9 (~340k seqs × 65,536 features × 4 B = ~88 GB) the prior
        # "build a list then np.array(list, dtype=...)" approach doubled
        # the peak host memory during construction (list + destination
        # array live simultaneously) and OOM-killed the training process
        # even with --mem=250G. Pre-allocating keeps peak at the single
        # 88 GB destination.
        self.features = np.empty((len(sequences), processor.num_features), dtype=np.float32)
        self._features_filled = 0
        self.sequence_ids = []
        self.raw_sequences = []
        self.length_aug_p = float(length_aug_p)
        self.length_aug_min = int(length_aug_min)
        self.length_aug_max = int(length_aug_max)
        
        self.domain_labels = []
        self.phylum_labels = []
        self.class_labels = []
        self.order_labels = []
        self.family_labels = []
        self.genus_labels = []
        self.species_labels = []
        
        self.domain_encoder = domain_encoder or {}
        self.phylum_encoder = phylum_encoder or {}
        self.class_encoder = class_encoder or {}
        self.order_encoder = order_encoder or {}
        self.family_encoder = family_encoder or {}
        self.genus_encoder = genus_encoder or {}
        self.species_encoder = species_encoder or {}
        
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        
        self.taxa_frequencies = defaultdict(int)
        
        self.taxonomy_hierarchy = taxonomy_hierarchy or TaxonomyHierarchy()
        
        if label_encoder is not None:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = {}
        
        create_encoder = (not any([
            domain_encoder, phylum_encoder, class_encoder,
            order_encoder, family_encoder, genus_encoder,
            species_encoder
        ]))
        self._encode_sequences(create_encoder=create_encoder)
        
        if taxonomy_hierarchy is None:
            self._build_taxonomy_hierarchy()
            self.taxonomy_hierarchy.log_statistics()
        
        # features already pre-allocated + filled in _encode_sequences; nothing to do here.
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
        if torch.is_tensor(idx):
            idx = idx.item()
        elif isinstance(idx, np.ndarray):
            idx = idx.item()

        if self.length_aug_p > 0.0 and torch.rand(1).item() < self.length_aug_p:
            seq = self.raw_sequences[idx]
            seq_len = len(seq)
            l_max = min(self.length_aug_max, seq_len)
            l_min = min(self.length_aug_min, l_max)
            if l_max > l_min:
                L = int(torch.randint(l_min, l_max + 1, (1,)).item())
                max_start = seq_len - L
                start = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
                truncated = seq[start:start + L]
                feature = torch.tensor(
                    self.processor.generate_kmer_frequencies(truncated),
                    dtype=torch.float32,
                )
            else:
                feature = torch.tensor(self.features[idx], dtype=torch.float32)
        else:
            feature = torch.tensor(self.features[idx], dtype=torch.float32)

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
        self._track_taxa_frequencies()

        if create_encoder:
            all_domains = set()
            all_phyla = set()
            all_classes = set()
            all_orders = set()
            all_families = set()
            all_genera = set()
            all_species = set()
            
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
            
            sorted_domains = sorted(list(all_domains))
            sorted_phyla = sorted(list(all_phyla))
            sorted_classes = sorted(list(all_classes))
            sorted_orders = sorted(list(all_orders))
            sorted_families = sorted(list(all_families))
            sorted_genera = sorted(list(all_genera))
            sorted_species = sorted(list(all_species))
            
            for sorted_list in [sorted_domains, sorted_phyla, sorted_classes, sorted_orders, 
                              sorted_families, sorted_genera, sorted_species]:
                if "unknown" in sorted_list:
                    sorted_list.remove("unknown")
                sorted_list.insert(0, "unknown")
            
            self.domain_encoder = {domain: idx for idx, domain in enumerate(sorted_domains)}
            self.phylum_encoder = {phylum: idx for idx, phylum in enumerate(sorted_phyla)}
            self.class_encoder = {class_name: idx for idx, class_name in enumerate(sorted_classes)}
            self.order_encoder = {order: idx for idx, order in enumerate(sorted_orders)}
            self.family_encoder = {family: idx for idx, family in enumerate(sorted_families)}
            self.genus_encoder = {genus: idx for idx, genus in enumerate(sorted_genera)}
            self.species_encoder = {species: idx for idx, species in enumerate(sorted_species)}

        for i, (seq_id, sequence, taxonomy) in enumerate(self.sequences):
            self.sequence_ids.append(seq_id)
            self.raw_sequences.append(sequence)

            feats = self.processor.generate_kmer_frequencies(sequence)
            self.features[i] = feats
            self._features_filled = i + 1
            
            domain = taxonomy.get("domain", "unknown")
            phylum = taxonomy.get("phylum", "unknown")
            class_name = taxonomy.get("class", "unknown")
            order = taxonomy.get("order", "unknown")
            family = taxonomy.get("family", "unknown")
            genus = taxonomy.get("genus", "unknown")
            species = taxonomy.get("species", "unknown")
            
            domain_idx = self.domain_encoder.get(domain, 0)
            phylum_idx = self.phylum_encoder.get(phylum, 0)
            class_idx = self.class_encoder.get(class_name, 0)
            order_idx = self.order_encoder.get(order, 0)
            family_idx = self.family_encoder.get(family, 0)
            genus_idx = self.genus_encoder.get(genus, 0)
            species_idx = self.species_encoder.get(species, 0)
            
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
    
    def _build_taxonomy_hierarchy(self):
        """Build the taxonomy hierarchy from the dataset's sequences."""
        
        for _, _, taxonomy in self.sequences:
            if not all(level in taxonomy for level in self.level_order):
                continue
                
            self.taxonomy_hierarchy.add_taxonomy(taxonomy)
    
    def get_taxonomy_hierarchy(self) -> TaxonomyHierarchy:
        """Get the taxonomy hierarchy object."""
        return self.taxonomy_hierarchy
        
    def get_consistent_taxonomy(self, predictions: Dict[str, Tuple[int, float]]) -> Dict[str, Tuple[int, float]]:

        taxonomy = {}
        for level in self.level_order:
            idx, _ = predictions.get(level, (0, 0.0))
            if level == "domain":
                name = self.get_domain_label(idx)
            elif level == "phylum":
                name = self.get_phylum_label(idx)
            elif level == "class":
                name = self.get_class_label(idx)
            elif level == "order":
                name = self.get_order_label(idx)
            elif level == "family":
                name = self.get_family_label(idx)
            elif level == "genus":
                name = self.get_genus_label(idx)
            elif level == "species":
                name = self.get_species_label(idx)
            else:
                name = "unknown"
            taxonomy[level] = name
        
        consistent_taxonomy = self.taxonomy_hierarchy.get_consistent_path(taxonomy)
        
        consistent_predictions = {}
        for level in self.level_order:
            name = consistent_taxonomy.get(level, "unknown")
            if name == "unknown":
                consistent_predictions[level] = (0, 0.0)
                continue
                
            if level == "domain":
                idx = self.domain_encoder.get(name, 0)
            elif level == "phylum":
                idx = self.phylum_encoder.get(name, 0)
            elif level == "class":
                idx = self.class_encoder.get(name, 0)
            elif level == "order":
                idx = self.order_encoder.get(name, 0)
            elif level == "family":
                idx = self.family_encoder.get(name, 0)
            elif level == "genus":
                idx = self.genus_encoder.get(name, 0)
            elif level == "species":
                idx = self.species_encoder.get(name, 0)
            else:
                idx = 0
                
            confidence = predictions.get(level, (0, 1.0))[1]
            consistent_predictions[level] = (idx, confidence)
        
        return consistent_predictions
