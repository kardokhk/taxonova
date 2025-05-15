import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Union, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from deeptaxonomist.model.classifier import Classifier

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """Optimize confidence thresholds for taxonomic levels based on metrics with 'Unknown' classifications."""
    
    def __init__(self, model: Classifier, 
                dataloader: DataLoader, 
                label_encoders: Dict[str, Dict[str, int]],
                initial_thresholds: Dict[str, float],
                device: torch.device,
                training_taxa: Dict[str, Set[str]] = None):
        """Initialize threshold optimizer.
        
        Args:
            model: Trained model to optimize thresholds for
            dataloader: Validation data for threshold optimization
            label_encoders: Dictionary of label encoders for each taxonomic level
            initial_thresholds: Starting thresholds for each level
            device: Computation device
            training_taxa: Dictionary mapping levels to taxa seen during training
        """
        self.model = model
        self.dataloader = dataloader
        self.label_encoders = label_encoders
        self.initial_thresholds = initial_thresholds
        self.device = device
        self.training_taxa = training_taxa
        
        # Convert label encoders for easier lookups
        self.id_to_label = {
            level: {idx: label for label, idx in encoders.items()}
            for level, encoders in (label_encoders or {}).items()
        }
        
        # Define evaluation levels
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        self.taxonomy_dims = {level: len(encoders) for level, encoders in label_encoders.items()}
        
        # Results storage
        self.collected_data = {}
        self.optimized_thresholds = {}
        
    def collect_prediction_data(self) -> Dict[str, List]:
        """Collect prediction data for all taxonomic levels with confidence scores."""
        if self.collected_data:
            return self.collected_data
            
        logger.info("Collecting prediction data for threshold optimization...")
        self.model.eval()
        
        # Structure to store predictions, true labels, and confidence scores
        result_data = {level: {
            "true_labels": [],
            "predictions": [],
            "confidences": [],
            "is_unseen": []
        } for level in self.taxonomy_dims}
        
        with torch.no_grad():
            for batch_idx, ((features, sequence_ids), labels) in enumerate(self.dataloader):
                features = features.to(self.device)
                batch_labels = {
                    level: labels[level].to(self.device)
                    for level in self.taxonomy_dims.keys()
                }
                
                outputs = self.model(features)
                
                # Process each sample in the batch
                for sample_idx in range(len(features)):
                    # Build predicted taxonomy for this sample
                    sample_taxonomy = {}
                    sample_confidences = {}
                    
                    # First pass: Get raw predictions for each level
                    for level in self.taxonomy_dims.keys():
                        if level not in outputs["logits"]:
                            continue
                        
                        logits = outputs["logits"][level][sample_idx:sample_idx+1]
                        confidence = outputs["confidence"][level][sample_idx].item()
                        
                        # Get prediction and probability
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                        pred = logits.argmax(dim=1).cpu().numpy()[0]
                        
                        # Use maximum softmax probability directly (calibrated approach)
                        combined_conf = probs.max()
                        
                        # Convert to label name
                        pred_name = self.id_to_label.get(level, {}).get(pred, str(pred))
                        
                        # Store prediction and confidence
                        sample_taxonomy[level] = pred_name
                        sample_confidences[level] = combined_conf
                    
                    # Second pass: Process metrics with validated taxonomy
                    for level in self.taxonomy_dims.keys():
                        if level not in outputs["logits"]:
                            continue
                        
                        true_label = batch_labels[level][sample_idx].cpu().numpy().item()
                        true_name = self.id_to_label.get(level, {}).get(true_label, str(true_label))
                        pred_name = sample_taxonomy[level]
                        combined_conf = sample_confidences[level]
                        
                        # Check if this taxon is unseen in training data
                        is_unseen = self.training_taxa and true_name not in self.training_taxa.get(level, set())
                        
                        # Store data for threshold optimization
                        result_data[level]["true_labels"].append(true_name)
                        result_data[level]["predictions"].append(pred_name)
                        result_data[level]["confidences"].append(combined_conf)
                        result_data[level]["is_unseen"].append(is_unseen)
        
        self.collected_data = result_data
        
        # Print basic stats
        for level in self.taxonomy_dims:
            total = len(result_data[level]["true_labels"])
            unseen = sum(result_data[level]["is_unseen"])
            logger.info(f"Level {level}: {total} total samples, {unseen} unseen ({unseen/total*100:.1f}%)")
            
        return result_data
        
    def evaluate_threshold(self, level: str, threshold: float, 
                          optimize_for: str = "f1") -> float:
        """Evaluate a specific threshold for a given taxonomic level.
        
        Args:
            level: Taxonomic level to evaluate
            threshold: Confidence threshold to evaluate
            optimize_for: Metric to optimize ('f1', 'precision', 'recall', or 'balanced')
            
        Returns:
            Score value for the specified metric at the given threshold
        """
        if not self.collected_data:
            self.collect_prediction_data()
            
        data = self.collected_data[level]
        true_labels = data["true_labels"]
        predictions = data["predictions"]
        confidences = data["confidences"]
        
        # Apply threshold to get final predictions
        final_predictions = [
            pred if conf >= threshold and pred != "Unclassified" else "Unknown"
            for pred, conf in zip(predictions, confidences)
        ]
        
        # Calculate metrics (treating "Unknown" as a valid class)
        precision = precision_score(true_labels, final_predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, final_predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, final_predictions, average='macro', zero_division=0)
        
        # Choose which metric to return based on optimization target
        if optimize_for == "f1":
            return f1
        elif optimize_for == "precision":
            return precision
        elif optimize_for == "recall":
            return recall
        elif optimize_for == "balanced":
            # Return a balanced combination of precision and recall
            return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            raise ValueError(f"Unknown optimization target: {optimize_for}")
        
    def grid_search(self, level: str, 
                   thresholds: Optional[List[float]] = None,
                   optimize_for: str = "f1") -> Tuple[float, float]:
        """Perform grid search to find optimal threshold for a level.
        
        Args:
            level: Taxonomic level to optimize
            thresholds: List of threshold values to evaluate, or None for defaults
            optimize_for: Metric to optimize for
            
        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)  # 0.1, 0.2, ..., 0.9
        
        best_threshold = 0.5  # Default
        best_score = 0.0
        
        logger.info(f"Grid search for {level} optimizing for {optimize_for}")
        
        # Evaluate each threshold
        results = []
        for threshold in thresholds:
            score = self.evaluate_threshold(level, threshold, optimize_for)
            results.append((threshold, score))
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Log results
        for threshold, score in sorted(results, key=lambda x: x[0]):
            logger.info(f"  Threshold {threshold:.2f}: {score:.4f}" + 
                      (" *" if threshold == best_threshold else ""))
        
        logger.info(f"Best threshold for {level}: {best_threshold:.2f} with {optimize_for}={best_score:.4f}")
        return best_threshold, best_score
    
    def optimize_thresholds(self, optimize_for: str = "f1", 
                          threshold_range: Optional[List[float]] = None) -> Dict[str, float]:
        """Optimize thresholds for all taxonomic levels.
        
        Args:
            optimize_for: Metric to optimize for
            threshold_range: Range of thresholds to evaluate
            
        Returns:
            Dictionary mapping taxonomic levels to optimal thresholds
        """
        if not self.collected_data:
            self.collect_prediction_data()
        
        optimized = {}
        
        for level in self.taxonomy_dims.keys():
            optimal_threshold, best_score = self.grid_search(
                level, 
                thresholds=threshold_range,
                optimize_for=optimize_for
            )
            optimized[level] = optimal_threshold
        
        self.optimized_thresholds = optimized
        return optimized
    
    def get_performance_curves(self, level: str) -> Dict[str, List[float]]:
        """Generate precision, recall, and F1 curves across threshold values.
        
        Args:
            level: Taxonomic level to generate curves for
            
        Returns:
            Dictionary with precision, recall, and F1 values across thresholds
        """
        if not self.collected_data:
            self.collect_prediction_data()
            
        thresholds = np.linspace(0.05, 0.95, 19)  # More granular for curves
        
        curves = {
            "thresholds": thresholds.tolist(),
            "precision": [],
            "recall": [],
            "f1": []
        }
        
        for threshold in thresholds:
            precision = self.evaluate_threshold(level, threshold, "precision")
            recall = self.evaluate_threshold(level, threshold, "recall")
            f1 = self.evaluate_threshold(level, threshold, "f1")
            
            curves["precision"].append(precision)
            curves["recall"].append(recall)
            curves["f1"].append(f1)
        
        return curves

class MetricsEvaluator:
    """Evaluator for model performance metrics across taxonomic levels, including 'Unknown' classifications."""
    
    def __init__(self, taxonomy_dims: Dict[str, int], device: torch.device, 
                 label_encoders: Dict[str, Dict[str, int]] = None,
                 confidence_thresholds: Dict[str, float] = None):
        """Initialize metrics evaluator.
        
        Args:
            taxonomy_dims: Dictionary mapping levels to number of classes
            device: Device to run computations on
            label_encoders: Dictionary of label encoders for each taxonomic level
            confidence_thresholds: Dictionary mapping taxonomic levels to confidence thresholds
        """
        self.taxonomy_dims = taxonomy_dims
        self.device = device
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        self.label_encoders = label_encoders
        self.id_to_label = {
            level: {idx: label for label, idx in encoders.items()}
            for level, encoders in (label_encoders or {}).items()
        }
        
        # Use provided confidence thresholds or initialize with defaults
        self.confidence_thresholds = confidence_thresholds or {
            'domain': 0.5,
            'phylum': 0.5,
            'class': 0.5,
            'order': 0.5,
            'family': 0.5,
            'genus': 0.5,
            'species': 0.5
        }
        
        # Track seen taxa in training set
        self.seen_taxa = {level: set() for level in self.taxonomy_dims.keys()}
        
        logger.debug(f"MetricsEvaluator initialized with device: {device}")
    
    def evaluate_metrics(self, model: Classifier, 
                        dataloader: DataLoader, training_taxa: Dict[str, Set[str]] = None) -> Dict[str, Dict[str, Union[float, Dict]]]:
        """Calculate precision, recall, and F1 score for each level, including 'Unknown' classifications."""
        if not isinstance(model, Classifier):
            raise TypeError("Model must be a Classifier")
            
        if model.training:
            raise RuntimeError("Model must be in eval mode for evaluation")
        
        if not dataloader or len(dataloader) == 0:
            raise ValueError("DataLoader cannot be empty")
        
        try:
            metrics = {level: defaultdict(list) for level in self.taxonomy_dims.keys()}
            
            # Track per-taxa classifications
            classifications = {level: defaultdict(lambda: {
                "total": 0, 
                "correct": 0, 
                "misclassified_to": defaultdict(int),
                "seen_correct": 0,
                "seen_incorrect": 0,
                "is_unseen": False,
                "low_confidence": 0,
                "seen_total": 0,
                "unseen_total": 0,
                "unseen_correct_unknown": 0,
                "unseen_incorrect": 0,
                "unseen_low_confidence": 0,
                "confidences": []
            }) for level in self.taxonomy_dims.keys()}
            
            with torch.no_grad():
                for (features, sequence_ids), labels in dataloader:
                    features = features.to(self.device)
                    batch_labels = {
                        level: labels[level].to(self.device)
                        for level in self.taxonomy_dims.keys()
                    }
                    
                    outputs = model(features)
                    
                    # Process each sample in the batch
                    for sample_idx in range(len(features)):
                        # Build predicted taxonomy for this sample
                        sample_taxonomy = {}
                        sample_confidences = {}
                        
                        # First pass: Get raw predictions for each level
                        for level in self.taxonomy_dims.keys():
                            if level not in outputs["logits"]:
                                continue
                            
                            logits = outputs["logits"][level][sample_idx:sample_idx+1]
                            confidence = outputs["confidence"][level][sample_idx].item()
                            
                            # Use softmax probabilities directly
                            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                            pred = logits.argmax(dim=1).cpu().numpy()[0]
                            
                            # Use maximum softmax probability
                            combined_conf = probs.max()
                            
                            # Convert to label name
                            pred_name = self.id_to_label.get(level, {}).get(pred, str(pred))
                            
                            # Store prediction and confidence
                            sample_taxonomy[level] = pred_name
                            sample_confidences[level] = combined_conf
                                
                        # Second pass: Process metrics
                        for level in self.taxonomy_dims.keys():
                            if level not in outputs["logits"]:
                                continue
                            
                            true_label = batch_labels[level][sample_idx].cpu().numpy().item()
                            true_name = self.id_to_label.get(level, {}).get(true_label, str(true_label))
                            pred_name = sample_taxonomy[level]
                            combined_conf = sample_confidences[level]
                            threshold = self.confidence_thresholds[level]
                            
                            # Check if this taxon is unseen in training data
                            is_unseen = training_taxa and true_name not in training_taxa.get(level, set())
                            
                            # Apply confidence threshold
                            is_confident = combined_conf >= threshold
                            final_pred = pred_name if is_confident and pred_name != "Unclassified" else "Unknown"
                            
                            # Store metrics
                            metrics[level]["true_labels"].append(true_name)
                            metrics[level]["pred_labels"].append(final_pred)
                            metrics[level]["raw_pred_labels"].append(pred_name)  # Store raw predictions without Unknown substitution
                            metrics[level]["confidences"].append(combined_conf)
                            metrics[level]["is_unseen"].append(is_unseen)
                
            final_metrics = {}
            for level, level_metrics in metrics.items():
                threshold = self.confidence_thresholds[level]
                
                # Process classification stats for detailed analysis
                classifications = defaultdict(lambda: {
                "total": 0,
                "correct": 0,
                "misclassified_to": defaultdict(int),
                "is_unseen": False,
                "low_confidence": 0,
                "seen_total": 0,
                "unseen_total": 0,
                "unseen_correct_unknown": 0,
                "unseen_incorrect": 0,
                "unseen_low_confidence": 0,
                "seen_correct": 0,
                "seen_incorrect": 0,
                "confidences": []
                })
                
                for true_name, pred_name, conf, is_unseen in zip(
                    level_metrics["true_labels"],
                    level_metrics["pred_labels"],
                    level_metrics["confidences"],
                    level_metrics["is_unseen"]
                ):
                    stats = classifications[true_name]
                    stats["total"] += 1
                    stats["confidences"].append(conf)

                    # Track seen/unseen totals
                    if is_unseen:
                        stats["is_unseen"] = True
                        stats["unseen_total"] += 1
                        if conf < threshold:
                            stats["unseen_low_confidence"] += 1
                    else:
                        stats["seen_total"] += 1

                    # Track confidence threshold
                    if conf < threshold:
                        stats["low_confidence"] += 1
                        pred_name = "Unknown"  # Force unknown for low confidence

                    # Track classification results
                    if pred_name == "Unknown":
                        if is_unseen:  # Correctly identified as unknown for unseen taxa
                            stats["correct"] += 1
                            stats["unseen_correct_unknown"] += 1
                        # Note: seen taxa marked as unknown are counted in low_confidence
                    elif true_name == pred_name:
                        stats["correct"] += 1  # Count per sequence
                        if not is_unseen:
                            stats["seen_correct"] += 1  # Track seen correct sequences explicitly
                    else:
                        stats["misclassified_to"][pred_name] += 1
                        if is_unseen:
                            stats["unseen_incorrect"] += 1
                        else:
                            stats["seen_incorrect"] += 1  # Track seen incorrect sequences explicitly
                
                # Calculate metrics (treating "Unknown" as a valid class)
                metrics_dict = self._calculate_metrics(level_metrics)
                
                # Calculate confusion matrix
                confusion_mat = self._calculate_confusion_matrix(level_metrics)
                
                # Calculate unseen taxa metrics
                unseen_metrics = self._calculate_unseen_taxa_metrics(level_metrics)
                
                final_metrics[level] = {
                    "precision": metrics_dict["precision"],
                    "recall": metrics_dict["recall"],
                    "f1": metrics_dict["f1"],
                    
                    # Store confusion matrix and other detailed metrics
                    "confusion_matrix": confusion_mat,
                    "classifications": dict(classifications),
                    "unseen_taxa_metrics": unseen_metrics,
                    "threshold": self.confidence_thresholds[level]
                }
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise RuntimeError(f"Error computing metrics: {str(e)}")
    
    def _calculate_metrics(self, level_metrics: Dict) -> Dict:
        """Calculate metrics including Unknown as a valid class.
        
        Args:
            level_metrics: Metrics for a specific taxonomic level
            
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        # Get predicted and true labels
        true_list = level_metrics["true_labels"]
        pred_list = level_metrics["pred_labels"]
        
        # Custom calculation of precision, recall, and F1 that treats "Unknown" predictions specially
        # Following the requirement:
        # - If a classifier refuses to classify a sequence (i.e., returns "Unknown"), 
        #   it is not counted as a TP or FP for precision calculation
        # - Unclassified sequences that should have been classified count as FN, lowering recall
        
        # Initialize counters
        tp = 0
        fp = 0
        fn = 0
        
        # Get all unique labels (excluding "Unknown")
        unique_true_labels = set([label for label in true_list if label != "Unknown"])
        unique_pred_labels = set([label for label in pred_list if label != "Unknown"])
        all_labels = sorted(unique_true_labels.union(unique_pred_labels))
        
        # Count true positives, false positives, and false negatives for each class
        for label in all_labels:
            for true_label, pred_label in zip(true_list, pred_list):
                # True positives: cases where both true and predicted labels match
                if true_label == label and pred_label == label:
                    tp += 1
                
                # False positives: cases where prediction is this label but true label is different
                # Only count for classified sequences (where pred_label is not "Unknown")
                elif pred_label == label and true_label != label:
                    fp += 1
                
                # False negatives: cases where true label is this label but prediction is different
                # This includes when the prediction is "Unknown" but should have been this label
                elif true_label == label and pred_label != label:
                    fn += 1
        
        # Compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _calculate_confusion_matrix(self, level_metrics: Dict) -> Dict:
        """Calculate confusion matrix including Unknown predictions.
        
        Args:
            level_metrics: Metrics for a specific taxonomic level
            
        Returns:
            Dictionary with confusion matrix information
        """
        true_labels = level_metrics["true_labels"]
        pred_labels = level_metrics["pred_labels"]
        
        # Get unique classes including "Unknown"
        unique_labels = sorted(set(true_labels) | set(pred_labels))
        
        # Calculate confusion matrix
        cm = confusion_matrix(
            true_labels, 
            pred_labels, 
            labels=unique_labels
        )
        
        return {
            "matrix": cm.tolist(),
            "labels": unique_labels
        }
    
    def _calculate_unseen_taxa_metrics(self, level_metrics: Dict) -> Dict:
        """Calculate metrics specific to unseen taxa.
        
        Args:
            level_metrics: Metrics for a specific taxonomic level
            
        Returns:
            Dictionary with unseen taxa metrics
        """
        # Calculate unseen taxa metrics
        total_unseen = sum(1 for is_unseen in level_metrics["is_unseen"] if is_unseen)
        unseen_correct = sum(1 for t, p, is_unseen in 
                           zip(level_metrics["true_labels"], level_metrics["pred_labels"], level_metrics["is_unseen"]) 
                           if is_unseen and (p == "Unknown" or t == p))
        
        unseen_confidences = [conf for conf, is_unseen in 
                            zip(level_metrics["confidences"], level_metrics["is_unseen"]) 
                            if is_unseen]
        
        avg_confidence = np.mean(unseen_confidences) if unseen_confidences else 0.0
        
        return {
            "total_count": total_unseen,
            "correct_unknown": unseen_correct,
            "avg_confidence": avg_confidence
        }
    
    def compute_cv_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Compute average metrics and standard deviations across CV folds.
        
        Args:
            fold_metrics: List of metric dictionaries from each fold
            
        Returns:
            Dictionary containing average metrics and standard deviations
        """
        avg_metrics = {}
        std_dev = {}
        
        # Standard metrics
        metric_keys = ["precision", "recall", "f1"]
        
        for level in self.taxonomy_dims.keys():
            avg_metrics[level] = {
                metric: np.mean([fold[level][metric] for fold in fold_metrics if metric in fold[level]])
                for metric in metric_keys
            }
            
            std_dev[level] = {
                metric: np.std([fold[level][metric] for fold in fold_metrics if metric in fold[level]])
                for metric in metric_keys
            }
        
        return {
            "average_metrics": avg_metrics,
            "std_dev": std_dev
        }
