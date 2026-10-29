"""
Post-prediction processing utilities for taxonomic classification.
"""
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from taxonova.core.thresholds import GRADED_SPECIES, scaled_profile

logger = logging.getLogger(__name__)

class TaxonomicConsistencyProcessor:
    
    def __init__(self, dataset, level_order: List[str]):
        self.dataset = dataset
        self.level_order = level_order
        
    def apply_consistency(self, predictions: Dict, sequences: List[Tuple]) -> Dict:
        consistent_predictions = {}
        consistent_predictions['predictions'] = {}
        consistent_predictions['confidence'] = {}
        # tx_wip_04 iter-B: pass through the per-row OOD score if the predictor
        # computed one (energy_phylum etc.). Consistency enforcement only
        # rewrites taxa/confidence, not the OOD score.
        if 'ood_score' in predictions:
            consistent_predictions['ood_score'] = predictions['ood_score']

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
        if 'ood_score' in predictions:
            result['ood_score'] = predictions['ood_score']

        for level in self.level_order:
            result[f'{level}_predictions'] = predictions['predictions'][level]
            result[f'{level}_confidence'] = predictions['confidence'][level]

        return result
    
    def write_tsv_output(self, args: Any, predictions: Dict, sequences: List[Tuple]) -> None:
        """Write classification results to TSV file.

        Hierarchical confidence monotonicity is enforced at write time: each
        rank's reported confidence is capped at its parent rank's reported
        confidence so that domain ≥ phylum ≥ class ≥ order ≥ family ≥ genus
        ≥ species, as required by the laws of probability for a hierarchical
        taxonomy (a sequence's species probability cannot exceed its genus
        probability since each species belongs to exactly one genus). The
        per-rank softmax heads in the model are independent and do not
        enforce this constraint, so we apply it post-hoc here. This also
        guarantees that if a parent rank's confidence falls below the
        threshold, all child ranks are marked Unclassified — uncertainty
        propagates downward.
        """
        try:
            ood_config = getattr(args, "ood_config", None)
            ood_n_relaxed = 0
            ood_n_strict = 0
            with open(args.output, 'w') as f:
                header = ["sequence_id"]
                for level in self.level_order:
                    header.extend([level, f"{level}_confidence"])
                f.write("\t".join(header) + "\n")

                strict_thresholds = self._get_level_thresholds(args)
                if ood_config is not None:
                    from taxonova.calibration.ood import gating_thresholds
                    ood_row_thresholds = gating_thresholds(
                        strict_thresholds, ood_config.get("relaxed_thresholds", {}))

                for idx, seq in enumerate(sequences):
                    row = [seq[0]]

                    # tx_wip_03 iter-2: per-input OOD-aware adaptive
                    # thresholding. Default score is phylum-MSP. tx_wip_04
                    # iter-B added the energy_phylum alternative; in that
                    # mode the per-row score lives in predictions['ood_score']
                    # (precomputed by the predictor) and the OOD direction
                    # flips (high energy = OOD).
                    if ood_config is not None:
                        score_method = ood_config.get("score_method", "phylum_msp")
                        if score_method == "phylum_msp":
                            score_rank = ood_config["score_rank"]
                            score = float(predictions['confidence'].get(score_rank, [1.0] * len(sequences))[idx])
                            is_ood_row = score < ood_config["tau"]
                        else:
                            from taxonova.calibration.ood import is_ood as _is_ood
                            score = float(predictions['ood_score'][idx])
                            is_ood_row = _is_ood(score, ood_config["tau"], score_method)
                        if is_ood_row:
                            # fix/ood-threshold-gating: never gate an OOD row
                            # below the user's thresholds. Releases 26.9.x
                            # used ood_config["relaxed_thresholds"] (0.0 at
                            # every rank) in place of them.
                            thresholds = ood_row_thresholds
                            ood_n_relaxed += 1
                        else:
                            thresholds = strict_thresholds
                            ood_n_strict += 1
                    else:
                        thresholds = strict_thresholds

                    parent_conf = 1.0
                    for level in self.level_order:
                        raw_confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        try:
                            raw_confidence = float(raw_confidence)
                        except (TypeError, ValueError):
                            raw_confidence = 0.0
                        # Monotonic cap: confidence at this rank cannot exceed parent.
                        confidence = min(raw_confidence, parent_conf)
                        parent_conf = confidence
                        threshold = thresholds.get(level, args.confidence_threshold)

                        if confidence < threshold:
                            row.extend(["Unclassified", f"{confidence:.4f}"])
                        else:
                            label = predictions['predictions'][level][idx]
                            if not label or label == "unknown":
                                label = "Unknown"
                            row.extend([label, f"{confidence:.4f}"])

                    f.write("\t".join(row) + "\n")

            if ood_config is not None:
                self._log_ood_summary(ood_config, ood_n_relaxed, ood_n_strict)
            logger.info(f"Successfully wrote classifications to {args.output}")
        except Exception as e:
            logger.error(f"Error writing classifications: {str(e)}")
            raise
    
    @staticmethod
    def _log_ood_summary(ood_config: Dict, n_flagged: int, n_strict: int) -> float:
        """Log the fraction of inputs flagged OOD; warn above
        OOD_FLAG_WARN_FRACTION or when the checkpoint's tau was fitted under the
        pre-fix id_quantile reading. Returns the flagged fraction."""
        from taxonova.calibration.ood import ID_QUANTILE_CONVENTION, OOD_FLAG_WARN_FRACTION
        total = n_flagged + n_strict
        frac = n_flagged / total if total else 0.0
        method = ood_config.get("score_method", "phylum_msp")
        logger.info(
            f"OOD flag: {n_flagged}/{total} ({100.0 * frac:.1f}%) inputs flagged out-of-distribution "
            f"(score_method={method}, tau={ood_config['tau']:.4f}); flagged inputs were gated at "
            f"max(requested, checkpoint OOD) thresholds, never below the requested thresholds"
        )
        if ood_config.get("id_quantile_convention") != ID_QUANTILE_CONVENTION:
            logger.warning(
                "This checkpoint's OOD tau was fitted before the id_quantile fix (taxonova <= 26.9.2); "
                "with energy_phylum and id_quantile=0.99 it flags about 99% of in-distribution reads. "
                "Calls are unaffected (requested thresholds are applied), but the OOD flag is not meaningful."
            )
        if frac > OOD_FLAG_WARN_FRACTION:
            logger.warning(
                f"{100.0 * frac:.1f}% of inputs were flagged out-of-distribution "
                f"(more than {100.0 * OOD_FLAG_WARN_FRACTION:.0f}%): the inputs may not match the "
                f"model's amplicon region or reference, or tau is miscalibrated."
            )
        return frac

    def _get_level_thresholds(self, args: Any) -> Dict[str, float]:
        """Per-rank gates for one run: the graded profile scaled by the
        requested species confidence, with --<level>_threshold overriding any
        rank explicitly.

        The recommended profile is 0.5 for domain through family, 0.7 for genus
        and 0.8 for species. Asking for a single species confidence scales the
        whole profile in that ratio, so the one number the user supplies means
        the same thing at every rank. Previously the parent gates were pinned at
        0.5 whatever was asked for, so `--confidence_threshold 0.8` still
        accepted a genus label at 0.5, and because abstention propagates
        downward a parent failing that fixed gate discarded an already-computed
        species call. Reviewer 2 raised both halves of this against the hosted
        service, where it is fixed; this brings the command-line tool into line,
        and matches Methods, "Architecture".

        Passing a rank explicitly opts out of the scaling for that rank, which
        is how a run reproduces a fixed profile regardless of this default.

        History: the flat 0.5 default arrived in tx_wip_02, when a threshold
        sweep showed a flat 0.8 rejecting confident-but-not-saturated
        predictions on OOD samples (coral / emp500_plant / emp500_soil) and
        collapsing recall (docs/threshold_sweep_results.md). Scaling a graded
        profile is not that flat 0.8: at the shipped default the parent ranks
        still sit at 0.5, which is the value that evidence supports, and only
        genus and species are raised.
        """
        result = {}
        for lvl, gate in scaled_profile(args.confidence_threshold).items():
            override = getattr(args, f"{lvl}_threshold", None)
            result[lvl] = gate if override is None else override
        result["species"] = args.confidence_threshold
        return result
