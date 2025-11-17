"""
Post-prediction processing utilities for taxonomic classification.
"""
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from taxonova.cli.logging_setup import console_log

logger = logging.getLogger(__name__)

class TaxonomicConsistencyProcessor:
    
    def __init__(self, dataset, level_order: List[str]):
        self.dataset = dataset
        self.level_order = level_order
        
    def apply_consistency(self, predictions: Dict, sequences: List[Tuple]) -> Dict:
        consistent_predictions = {}
        consistent_predictions['predictions'] = {}
        consistent_predictions['confidence'] = {}
        
        for level in self.level_order:
            consistent_predictions['predictions'][level] = []
            consistent_predictions['confidence'][level] = []
            
        for idx, seq in enumerate(sequences):
            current_pred = {}
            for level in self.level_order:
                label = predictions['predictions'][level][idx]
                conf = predictions['confidence'][level][idx]
                
                pred_idx = self._get_label_index(level, label)
                current_pred[level] = (pred_idx, conf)
            
            consistent_pred = self.dataset.get_consistent_taxonomy(current_pred)
            
            for level in self.level_order:
                pred_idx, conf = consistent_pred[level]
                label = self._get_index_label(level, pred_idx)
                
                consistent_predictions['predictions'][level].append(label)
                consistent_predictions['confidence'][level].append(conf)
        
        return consistent_predictions
    
    def _get_label_index(self, level: str, label: str) -> int:
        """Get index for a label at a given taxonomic level."""
        encoder_map = {
            "domain": self.dataset.domain_encoder,
            "phylum": self.dataset.phylum_encoder,
            "class": self.dataset.class_encoder,
            "order": self.dataset.order_encoder,
            "family": self.dataset.family_encoder,
            "genus": self.dataset.genus_encoder,
            "species": self.dataset.species_encoder
        }
        return encoder_map.get(level, {}).get(label, 0)
    
    def _get_index_label(self, level: str, pred_idx: int) -> str:
        """Get label for an index at a given taxonomic level."""
        label_func_map = {
            "domain": self.dataset.get_domain_label,
            "phylum": self.dataset.get_phylum_label,
            "class": self.dataset.get_class_label,
            "order": self.dataset.get_order_label,
            "family": self.dataset.get_family_label,
            "genus": self.dataset.get_genus_label,
            "species": self.dataset.get_species_label
        }
        return label_func_map.get(level, lambda x: "unknown")(pred_idx)

class PredictionOutputFormatter:
    """Formats prediction outputs for different use cases."""
    
    def __init__(self, level_order: List[str]):
        self.level_order = level_order
    
    def format_predictions(self, predictions: Dict) -> Dict:
        """Format predictions for output."""
        prediction_components = []
        num_predictions = len(predictions['predictions']['domain'])
        
        for i in range(num_predictions):
            components = {level: predictions['predictions'][level][i] 
                        for level in self.level_order}
            prediction_components.append(components)
        
        result = {
            'predictions': {level: [comp[level] for comp in prediction_components]
                          for level in self.level_order},
            'confidence': {level: predictions['confidence'][level]
                         for level in self.level_order}
        }
        
        for level in self.level_order:
            result[f'{level}_predictions'] = predictions['predictions'][level]
            result[f'{level}_confidence'] = predictions['confidence'][level]
        
        return result
    
    def write_tsv_output(self, args: Any, predictions: Dict, sequences: List[Tuple], config=None) -> None:
        """Write classification results to TSV file."""
        try:
            with open(args.output, 'w') as f:
                header = ["sequence_id"]
                for level in self.level_order:
                    header.extend([level, f"{level}_confidence"])
                f.write("\t".join(header) + "\n")
                
                for idx, seq in enumerate(sequences):
                    row = [seq[0]]
                    
                    thresholds = self._get_level_thresholds(config if config else args)
                    
                    for level in self.level_order:
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = thresholds.get(level, 0.5)
                        
                        if confidence < threshold:
                            row.extend(["Unclassified", f"{confidence:.4f}"])
                        else:
                            label = predictions['predictions'][level][idx]
                            if not label or label == "unknown":
                                label = "Unknown"
                            row.extend([label, f"{confidence:.4f}"])
                    
                    f.write("\t".join(row) + "\n")
                
            console_log(f"✓ Results written to {args.output}")
        except Exception as e:
            logger.error(f"Error writing classifications: {str(e)}")
            raise
    
    def _get_level_thresholds(self, source: Any) -> Dict[str, float]:
        """Get confidence thresholds for each taxonomic level."""
        if hasattr(source, 'domain_confidence_threshold'):
            return {
                "domain": source.domain_confidence_threshold,
                "phylum": source.phylum_confidence_threshold,
                "class": source.class_confidence_threshold,
                "order": source.order_confidence_threshold,
                "family": source.family_confidence_threshold,
                "genus": source.genus_confidence_threshold,
                "species": source.species_confidence_threshold
            }
        else:
            return {
                "domain": getattr(source, 'domain_confidence', 0.5),
                "phylum": getattr(source, 'phylum_confidence', 0.5),
                "class": getattr(source, 'class_confidence', 0.5),
                "order": getattr(source, 'order_confidence', 0.5),
                "family": getattr(source, 'family_confidence', 0.4),
                "genus": getattr(source, 'genus_confidence', 0.5),
                "species": getattr(source, 'species_confidence', 0.6)
            }
