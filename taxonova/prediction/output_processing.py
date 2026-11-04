"""
Post-prediction processing utilities for taxonomic classification.
"""
import logging
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

from taxonova.core.thresholds import GRADED_SPECIES, scaled_profile

RANKS7 = ("domain", "phylum", "class", "order", "family", "genus", "species")

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
        if 'no_signal' in predictions:
            consistent_predictions['no_signal'] = predictions['no_signal']
        if 'species_gate_stat' in predictions:
            consistent_predictions['species_gate_stat'] = predictions['species_gate_stat']

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

            # The consistency step rewrites parent labels from the deepest
            # named rank (normally species). A rewritten rank's confidence was
            # measured for the label the model predicted there, not for the
            # new one, so it is capped at the confidence of the rank that
            # drove the rewrite; unchanged ranks keep their own confidence.
            driver_conf = None
            for level in reversed(self.level_order):
                if predictions['predictions'][level][idx] not in (None, "", "unknown"):
                    driver_conf = float(predictions['confidence'][level][idx])
                    break

            for level in self.level_order:
                pred_idx, conf = consistent_pred[level]
                label = self._get_index_label(level, pred_idx)
                if driver_conf is not None and label != predictions['predictions'][level][idx]:
                    conf = min(float(conf), driver_conf)

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
        if 'no_signal' in predictions:
            result['no_signal'] = predictions['no_signal']
        if 'species_gate_stat' in predictions:
            result['species_gate_stat'] = predictions['species_gate_stat']

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
                from taxonova.core.operating_point import (gate_passes, low_information,
                                                           low_information_limits, row_thresholds)
                min_valid, max_amb = low_information_limits(getattr(args, "operating_point_entry", None))
                n_low_info = 0
                species_gate = getattr(args, "species_gate", None)
                gate_stats = predictions.get("species_gate_stat")
                if species_gate is not None and gate_stats is None:
                    raise ValueError("species gate configured but the predictor returned no gate statistic")
                n_gated = 0
                # species group labels reported at the genus (taxonova/species_groups.py), when the checkpoint asks
                group_collapse = getattr(args, "species_group_collapse", None)
                if group_collapse is not None and list(self.level_order) != list(RANKS7):
                    raise ValueError("species group collapse needs the seven ranks domain..species")
                n_collapsed = Counter()

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
                            ood_extra = ood_config.get("relaxed_thresholds", {})
                            ood_n_relaxed += 1
                        else:
                            ood_extra = None
                            ood_n_strict += 1
                    else:
                        ood_extra = None
                    sp_label = predictions['predictions']['species'][idx]
                    # an empty / 'unknown' species label is no species call: nothing for the gate to withhold
                    gate_ok = (species_gate is None or sp_label in ("", None, "unknown", "Unknown")
                               or gate_passes(species_gate, gate_stats[idx]))
                    n_gated += 0 if gate_ok else 1
                    # one place computes the row's gates, elementwise never below strict_thresholds
                    low_info = low_information(seq[1], min_valid, max_amb) if len(seq) > 1 else True
                    n_low_info += int(low_info)
                    thresholds = row_thresholds(strict_thresholds, ood_extra, gate_ok, low_info)

                    no_signal = predictions.get('no_signal')
                    if no_signal is not None and no_signal[idx]:
                        row.extend(["Unclassified", "0.0000"] * len(self.level_order))
                        f.write("\t".join(row) + "\n")
                        continue

                    parent_conf = 1.0
                    parent_called = True
                    cells = []
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

                        # Abstention propagates downward whatever the thresholds: with calibrated gates a child
                        # rank's threshold can be below its parent's (genus 0.38 under family 0.5), and the capped
                        # confidence alone would then name a genus under an unclassified family.
                        if not parent_called or confidence < threshold:
                            cells.append(["Unclassified", f"{confidence:.4f}"])
                            parent_called = False
                        else:
                            label = predictions['predictions'][level][idx]
                            if not label or label == "unknown":
                                label = "Unknown"
                            cells.append([label, f"{confidence:.4f}"])

                    if group_collapse is not None:
                        called = ["" if lab in ("Unclassified", "Unknown") else lab for lab, _ in cells]
                        kept, why = group_collapse(called)
                        n_collapsed[why] += 1
                        for cell, before, after in zip(cells, called, kept):
                            if before and not after:
                                cell[0] = "Unclassified"     # the confidence is still reported
                    for cell in cells:
                        row.extend(cell)
                    f.write("\t".join(row) + "\n")

            if ood_config is not None:
                self._log_ood_summary(ood_config, ood_n_relaxed, ood_n_strict)
            if n_low_info:
                logger.warning(f"{n_low_info}/{len(sequences)} inputs have fewer than {min_valid} valid bases or "
                               f"more than {100 * max_amb:.0f}% ambiguous characters: no call at any rank")
            if species_gate is not None:
                logger.info(f"Species gate ({species_gate['statistic']}): species call withheld for "
                            f"{n_gated}/{len(sequences)} inputs")
            if group_collapse is not None:
                logger.info("Species group labels (reported at the genus): "
                            + ", ".join(f"{k} {v}" for k, v in sorted(n_collapsed.items())))
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
        resolved = getattr(args, "level_thresholds", None)
        if resolved:
            return dict(resolved)
        result = {}
        for lvl, gate in scaled_profile(args.confidence_threshold).items():
            override = getattr(args, f"{lvl}_threshold", None)
            result[lvl] = gate if override is None else override
        result["species"] = args.confidence_threshold
        return result
