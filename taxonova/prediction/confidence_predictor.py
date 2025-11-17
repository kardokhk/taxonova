"""
Confidence-aware taxonomic prediction system.

"""
import torch
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ConfidenceAwareTaxonomicPredictor:
    
    def __init__(self, model, dataset, taxonomy_hierarchy, device: str = "cuda"):
        self.model = model
        self.dataset = dataset
        self.taxonomy_hierarchy = taxonomy_hierarchy
        self.device = device
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        
    def predict_sequences(self, sequences: List[Tuple], 
                         confidence_thresholds: Optional[Dict[str, float]] = None) -> Dict:

        if confidence_thresholds is None:
            confidence_thresholds = {level: 0.5 for level in self.level_order}
        
        self.model.eval()
        predictions = {level: [] for level in self.level_order}
        confidences = {level: [] for level in self.level_order}
        deepest_classifications = []
        
        with torch.no_grad():
            for seq_id, sequence in sequences:
                features = self.dataset.processor.generate_kmer_frequencies(sequence)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                result = self.model.predict_with_confidence_thresholds(
                    features_tensor, **confidence_thresholds
                )
                
                seq_predictions = {}
                seq_confidences = {}
                deepest_level = None
                
                for level in self.level_order:
                    pred_idx = result['predictions'][level][0].item()
                    confidence = result['confidences'][level][0].item()
                    
                    if pred_idx == 0:
                        label = "Unclassified"
                    else:
                        label = self._get_label_from_index(level, pred_idx)
                    
                    seq_predictions[level] = label
                    seq_confidences[level] = confidence
                    
                    if confidence >= confidence_thresholds[level] and label != "Unclassified":
                        deepest_level = level
                
                for level in self.level_order:
                    predictions[level].append(seq_predictions[level])
                    confidences[level].append(seq_confidences[level])
                
                deepest_classifications.append(deepest_level if deepest_level else "domain")
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'deepest_classifications': deepest_classifications,
            'confidence_thresholds': confidence_thresholds
        }
    
    def _get_label_from_index(self, level: str, index: int) -> str:
        """Convert taxonomic level index back to label."""
        label_func_map = {
            "domain": self.dataset.get_domain_label,
            "phylum": self.dataset.get_phylum_label,
            "class": self.dataset.get_class_label,
            "order": self.dataset.get_order_label,
            "family": self.dataset.get_family_label,
            "genus": self.dataset.get_genus_label,
            "species": self.dataset.get_species_label
        }
        return label_func_map[level](index)
    
    def evaluate_predictions(self, predictions: Dict, true_sequences: List[Tuple]) -> Dict:

        level_metrics = {}
        
        for level in self.level_order:
            true_labels = []
            pred_labels = []
            confidences = predictions['confidences'][level]
            
            for i, (_, _, taxonomy) in enumerate(true_sequences):
                true_label = taxonomy.get(level, "unknown")
                if true_label == "unknown" or not true_label.strip():
                    continue
                
                true_labels.append(true_label)
                pred_labels.append(predictions['predictions'][level][i])
            
            if len(true_labels) > 0:
                correct = sum(1 for t, p in zip(true_labels, pred_labels) 
                            if t == p and p != "Unclassified")
                total_with_true_labels = len(true_labels)
                accuracy = correct / total_with_true_labels if total_with_true_labels > 0 else 0.0
                
                unique_labels = set(true_labels) - {"unknown", ""}
                total_tp = total_fp = total_fn = 0
                
                for label in unique_labels:
                    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
                    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
                    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                level_metrics[level] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'num_sequences': total_with_true_labels
                }
            else:
                level_metrics[level] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'num_sequences': 0
                }
        
        total_f1 = sum(level_metrics[level]['f1'] for level in self.level_order)
        overall_f1 = total_f1 / len(self.level_order)
        
        depth_weighted_f1 = 0.0
        total_weight = 0.0
        for i, level in enumerate(self.level_order):
            weight = 1.0 + (i * 0.5)
            depth_weighted_f1 += weight * level_metrics[level]['f1']
            total_weight += weight
        depth_weighted_f1 = depth_weighted_f1 / total_weight if total_weight > 0 else 0.0
        
        return {
            'level_metrics': level_metrics,
            'overall_f1': overall_f1,
            'depth_weighted_f1': depth_weighted_f1,
            'deepest_classification_stats': self._analyze_deepest_classifications(predictions)
        }
    
    def _analyze_deepest_classifications(self, predictions: Dict) -> Dict:
        """Analyze the distribution of deepest confident classifications."""
        deepest_counts = {}
        for level in self.level_order + ['unclassified']:
            deepest_counts[level] = 0
        
        for deepest in predictions['deepest_classifications']:
            if deepest is None:
                deepest_counts['unclassified'] += 1
            else:
                deepest_counts[deepest] += 1
        
        total = sum(deepest_counts.values())
        deepest_percentages = {level: (count / total * 100) if total > 0 else 0.0 
                             for level, count in deepest_counts.items()}
        
        return {
            'counts': deepest_counts,
            'percentages': deepest_percentages
        }
