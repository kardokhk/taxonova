"""
Classification predictor
"""
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from taxonova.data.dataset import TaxonomyDataset
from taxonova.prediction.hierarchical_model import HierarchicalClassifier
from taxonova.prediction.device_manager import DeviceManager
from taxonova.utils.resource_tracking import track_resources

logger = logging.getLogger(__name__)

class HierarchicalPredictor:
    
    def __init__(self, model: HierarchicalClassifier, dataset: TaxonomyDataset, 
                 config, device: str = None):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        
        self.device = device if device is not None else 'cpu'
        self.model = self.model.to(self.device)
        
        self.domain_encoder = dataset.domain_encoder
        self.phylum_encoder = dataset.phylum_encoder
        self.class_encoder = dataset.class_encoder
        self.order_encoder = dataset.order_encoder
        self.family_encoder = dataset.family_encoder
        self.genus_encoder = dataset.genus_encoder
        self.species_encoder = dataset.species_encoder
    
    @track_resources("Prediction")
    def predict(self, sequences: List[Tuple], batch_size: int = None) -> Dict:
        self.model.eval()
        
        if batch_size is None:
            batch_size = self.config.batch_size * 2
            
        if hasattr(self.dataset, 'sequences') and len(sequences) == len(self.dataset.sequences):
            eval_dataset = self.dataset
        else:
            from taxonova.data.dataset import TaxonomyDataset
            eval_dataset = TaxonomyDataset(
                sequences=sequences,
                processor=self.dataset.processor,
                label_encoder=self.dataset.label_encoder if hasattr(self.dataset, 'label_encoder') else None
            )
            
            eval_dataset.domain_encoder = self.domain_encoder
            eval_dataset.phylum_encoder = self.phylum_encoder
            eval_dataset.class_encoder = self.class_encoder
            eval_dataset.order_encoder = self.order_encoder
            eval_dataset.family_encoder = self.family_encoder
            eval_dataset.genus_encoder = self.genus_encoder
            eval_dataset.species_encoder = self.species_encoder
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        predictions = {level: {'preds': [], 'confidences': []}
                      for level in self.level_order}
        # tx_wip_04 iter-B: when an alternate OOD score method is requested
        # (e.g. energy_phylum), accumulate the per-row score here so
        # write_tsv_output can compare it against τ at write time.
        ood_score_method = getattr(self.config, "ood_score_method", None)
        ood_scores = [] if ood_score_method else None

        with torch.no_grad():
            pred_pbar = tqdm(eval_loader, desc="Classifying", leave=False, unit="batch")
            for batch in pred_pbar:
                (features, _), _ = batch
                features = features.to(self.device)

                outputs = self.model(features)

                # Collect per-rank predictions + raw confidences first.
                batch_preds = {}
                batch_confs = {}
                for level in self.level_order:
                    logits = outputs[f"{level}_logits"]
                    confidence = outputs[f"{level}_confidence"]
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    batch_preds[level] = preds.cpu().numpy()
                    batch_confs[level] = confidence.cpu().numpy()

                if ood_scores is not None:
                    from taxonova.calibration.ood import compute_ood_score_from_outputs
                    ood_scores.extend(
                        compute_ood_score_from_outputs(outputs, ood_score_method).cpu().numpy().tolist()
                    )

                # Enforce hierarchical confidence monotonicity:
                # each rank's confidence is capped at its parent rank's
                # confidence (domain ≥ phylum ≥ ... ≥ species). This is
                # required because the per-rank softmax heads are independent
                # in the model architecture and so do not satisfy the
                # P(child) ≤ P(parent) constraint of a true hierarchical
                # probability model.
                parent_conf = None
                for level in self.level_order:
                    if parent_conf is None:
                        capped = batch_confs[level]
                    else:
                        capped = np.minimum(batch_confs[level], parent_conf)
                    parent_conf = capped
                    predictions[level]['preds'].extend(batch_preds[level])
                    predictions[level]['confidences'].extend(capped)
        
        for level in self.level_order:
            get_label_func = getattr(self.dataset, f'get_{level}_label')
            predictions[level]['labels'] = [get_label_func(pred) 
                                          for pred in predictions[level]['preds']]
        
        from taxonova.prediction.output_processing import PredictionOutputFormatter
        formatter = PredictionOutputFormatter(self.level_order)
        
        formatted_predictions = {
            'predictions': {level: predictions[level]['labels'] for level in self.level_order},
            'confidence': {level: predictions[level]['confidences'] for level in self.level_order}
        }
        if ood_scores is not None:
            formatted_predictions['ood_score'] = ood_scores

        return formatter.format_predictions(formatted_predictions)
