import itertools
import logging
import numpy as np
import torch
import os
import time
import json
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, List
import random
from taxonova.data.fasta import read_fasta_file
from taxonova.model.model_config import ModelConfig
from taxonova.model.model_factory import ModelFactory
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.data.dataset import TaxonomyDataset

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, train_db: str, base_config: ModelConfig, search_space: dict, mode: str = 'random',
                 n_iterations: int = 100, early_stopping_rounds: int = 10, verbose: bool = False):
        self.train_db = train_db
        self.base_config = base_config
        self.search_space = search_space
        self.mode = mode
        self.n_iterations = n_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.output_dir = os.path.join("output", "tuning_results", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        
    # ...existing code...
    
    def tune_weights(self, num_epochs: int = 5, folds: int = 5, min_delta: float = 0.001):
        """
        Tune the weights for hierarchical classification to optimize F1 score.
        This specifically tunes family_weight, genus_weight, species_weight.
        """
        if not self.base_config.hierarchical:
            raise ValueError("Weight tuning is only applicable for hierarchical classification")
            
        logger.info("Starting hierarchical weights tuning...")
        sequences = read_fasta_file(self.train_db)
        processor = SequenceProcessor()
        dataset = TaxonomyDataset(sequences=sequences, processor=processor, hierarchical=True)
        
        best_params = None
        best_f1 = -np.inf
        best_cv_metrics = None
        best_model = None
        best_visualization_dir = None
        
        # Early stopping tracking
        rounds_without_improvement = 0
        history = []
        
        # Define weight search space if it doesn't already include weight parameters
        weight_search_space = {
            'family_weight': {'range': [0.2, 1.5]},
            'genus_weight': {'range': [0.2, 1.5]},
            'species_weight': {'range': [0.5, 2.0]},
        }
        # Use provided search space if it already has the weight parameters
        if any(key in self.search_space for key in weight_search_space):
            weight_search_space = {k: v for k, v in self.search_space.items() 
                                  if k in ['family_weight', 'genus_weight', 'species_weight']}
        
        # Calculate total iterations based on mode
        if self.mode == 'random':
            total_trials = self.n_iterations
            desc = "Random Weight Search"
        else:  # Grid search
            # Create grid of weight combinations
            family_weights = np.linspace(0.2, 1.5, 5)
            genus_weights = np.linspace(0.2, 1.5, 5)
            species_weights = np.linspace(0.5, 2.0, 5)
            weight_combinations = list(itertools.product(family_weights, genus_weights, species_weights))
            total_trials = len(weight_combinations)
            desc = "Grid Weight Search"
        
        tune_pbar = tqdm(total=total_trials, desc=desc, unit="trial")
        
        # Generate weight combinations based on search mode
        def get_weight_combinations():
            if self.mode == 'grid':
                if all(isinstance(v, dict) and 'range' in v for v in weight_search_space.values()):
                    # Create evenly spaced grids for each weight
                    family_weights = np.linspace(
                        weight_search_space['family_weight']['range'][0],
                        weight_search_space['family_weight']['range'][1],
                        5
                    )
                    genus_weights = np.linspace(
                        weight_search_space['genus_weight']['range'][0],
                        weight_search_space['genus_weight']['range'][1],
                        5
                    )
                    species_weights = np.linspace(
                        weight_search_space['species_weight']['range'][0],
                        weight_search_space['species_weight']['range'][1],
                        5
                    )
                    
                    for f, g, s in itertools.product(family_weights, genus_weights, species_weights):
                        yield {
                            'family_weight': float(f),
                            'genus_weight': float(g),
                            'species_weight': float(s)
                        }
                else:
                    # Use the provided values directly
                    keys = list(weight_search_space.keys())
                    values = list(weight_search_space.values())
                    for combo in itertools.product(*values):
                        yield dict(zip(keys, combo))
            else:  # Random search
                for _ in range(self.n_iterations):
                    params = {}
                    for key, values in weight_search_space.items():
                        if isinstance(values, (list, tuple)):
                            params[key] = random.choice(values)
                        elif isinstance(values, dict) and 'range' in values:
                            min_val, max_val = values['range']
                            params[key] = np.random.uniform(min_val, max_val)
                    yield params
        
        # Test each weight combination
        for trial_idx, weight_params in enumerate(get_weight_combinations()):
            # Update the base config with the new weights
            config_dict = self.base_config.__dict__.copy()
            config_dict.update(weight_params)
            config = ModelConfig(**config_dict)
            
            # Round weights for logging
            log_params = {k: round(v, 3) if isinstance(v, float) else v 
                         for k, v in weight_params.items()}
            
            if self.verbose:
                logger.info(f"Testing weights: {log_params}")
            else:
                # Just log a simple message for which trial we're on
                logger.info(f"Running weight trial {trial_idx+1}/{total_trials}")
                
            trainer = ModelFactory.create_trainer(
                sequences=sequences,
                config=config,
                dataset=dataset,
                validation_mode="cv"
            )
            model, cv_metrics = trainer.cross_validate(num_epochs=num_epochs)
            
            # For hierarchical models, we use overall_f1 or combined F1 from family, genus, species
            if 'overall_f1' in cv_metrics['average_metrics']:
                f1_score = cv_metrics['average_metrics']['overall_f1'] 
            else:
                # Use weighted average if overall F1 isn't available
                family_f1 = cv_metrics['average_metrics'].get('family', {}).get('f1', 0)
                genus_f1 = cv_metrics['average_metrics'].get('genus', {}).get('f1', 0) 
                species_f1 = cv_metrics['average_metrics'].get('species', {}).get('f1', 0)
                f1_score = 0.2 * family_f1 + 0.3 * genus_f1 + 0.5 * species_f1
            
            # Update progress bar
            tune_pbar.set_postfix({
                'best_f1': f"{best_f1:.4f}",
                'curr_f1': f"{f1_score:.4f}"
            })
            tune_pbar.update(1)
            tune_pbar.set_description(f"{desc} (Best F1: {best_f1:.4f})")
            
            # Save detailed results
            trial_results = {
                'params': log_params,
                'f1': f1_score,
                'cv_metrics': cv_metrics
            }
            
            if self.verbose:
                logger.info(f"\nTrial Results:")
                logger.info(f"F1 Score: {f1_score:.4f}")
                logger.info(f"Parameters: {log_params}")
            
            # Track all results
            history.append(trial_results)
            
            improvement = f1_score - best_f1
            if improvement > min_delta:
                best_f1 = f1_score
                best_params = weight_params
                best_cv_metrics = cv_metrics
                best_model = model
                best_visualization_dir = getattr(trainer, 'last_visualization_dir', None)
                logger.info(f"New best weights found! F1 score improvement: {improvement:.4f}")
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                
            # Early stopping check
            if self.mode == 'random' and rounds_without_improvement >= self.early_stopping_rounds:
                logger.info(f"Early stopping triggered after {rounds_without_improvement} rounds without improvement")
                break
        
        # Export detailed results to files
        top_configs_file, trials_file = self.export_detailed_results(
            history, best_params, best_f1, best_visualization_dir
        )
        
        # Print summary
        logger.info(f"\nHierarchical weight tuning completed.")
        logger.info(f"Best weights:")
        for weight_name, weight_value in best_params.items():
            logger.info(f"  {weight_name}: {weight_value:.3f}")
        logger.info(f"Best F1 score: {best_f1:.4f}")
        logger.info(f"Top configurations saved to: {top_configs_file}")
        logger.info(f"All trial results saved to: {trials_file}")
        
        tune_pbar.close()
        
        return best_params, best_cv_metrics, best_model, best_visualization_dir

    def _get_random_params(self) -> Dict[str, Any]:
        """Sample random parameters from the search space."""
        params = {}
        for key, values in self.search_space.items():
            if isinstance(values, (list, tuple)):
                params[key] = random.choice(values)
            elif isinstance(values, dict) and 'range' in values:
                # Handle continuous ranges with log scale option
                min_val, max_val = values['range']
                if values.get('integer', False):
                    # For integer parameters
                    if values.get('log_scale', False):
                        params[key] = int(np.exp(np.random.uniform(np.log(min_val), np.log(max_val))))
                    else:
                        params[key] = int(np.random.uniform(min_val, max_val))
                else:
                    # For float parameters
                    if values.get('log_scale', False):
                        params[key] = np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
                    else:
                        params[key] = np.random.uniform(min_val, max_val)
        return params

    def _get_hyperparameter_combinations(self):
        if self.mode == 'grid':
            keys = list(self.search_space.keys())
            values = list(self.search_space.values())
            for combo in itertools.product(*values):
                yield dict(zip(keys, combo))
        elif self.mode == 'random':
            for _ in range(self.n_iterations):
                yield self._get_random_params()
    
    def export_detailed_results(self, history, best_params, best_f1, best_visualization_dir):
        """Export detailed results to files instead of printing to console."""
        # Export all trial results
        trials_file = os.path.join(self.output_dir, "all_trials.json")
        serializable_history = []
        for item in history:
            # Make the cv_metrics serializable
            clean_item = {
                'params': item['params'],
                'f1': item['f1']
            }
            serializable_history.append(clean_item)
            
        with open(trials_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        # Export top configurations
        top_n = min(5, len(history))
        sorted_history = sorted(history, key=lambda x: x['f1'], reverse=True)
        top_configs_file = os.path.join(self.output_dir, "top_configurations.txt")
        
        with open(top_configs_file, 'w') as f:
            f.write(f"Top {top_n} configurations:\n")
            for i, result in enumerate(sorted_history[:top_n], 1):
                f.write(f"{i}. F1 Score: {result['f1']:.4f}, Params: {str(result['params'])}\n")
            
            f.write(f"\nBest hyperparameters: {best_params} with F1 score: {best_f1:.4f}\n")
            f.write(f"Visualization folder for best model: {best_visualization_dir}\n")
        
        logger.info(f"Detailed tuning results exported to {self.output_dir}")
        return top_configs_file, trials_file
    
    def tune(self, num_epochs: int = 10, folds: int = 3, min_delta: float = 0.001):
        logger.info("Starting hyperparameter tuning...")
        sequences = read_fasta_file(self.train_db)
        processor = SequenceProcessor()
        dataset = TaxonomyDataset(sequences=sequences, processor=processor)
        
        best_params = None
        best_f1 = -np.inf
        best_cv_metrics = None
        best_model = None
        best_visualization_dir = None
        
        # Early stopping tracking
        rounds_without_improvement = 0
        history = []
        
        # Calculate total iterations and create main progress bar
        if self.mode == 'random':
            total_trials = self.n_iterations
            desc = "Random Search"
        else:
            total_trials = len(list(itertools.product(*self.search_space.values())))
            desc = "Grid Search"
        
        tune_pbar = tqdm(total=total_trials, desc=desc, unit="trial")
        
        for trial_idx, params in enumerate(self._get_hyperparameter_combinations()):
            # Update the base config with the new hyperparameters
            config_dict = self.base_config.__dict__.copy()
            config_dict.update(params)
            config = ModelConfig(**config_dict)
            
            # Round continuous parameters for logging
            log_params = {k: round(v, 6) if isinstance(v, float) else v 
                         for k, v in params.items()}
            
            if self.verbose:
                logger.info(f"Testing hyperparameters: {log_params}")
            else:
                # Just log a simple message for which trial we're on
                logger.info(f"Running trial {trial_idx+1}/{total_trials}")
                
            trainer = ModelFactory.create_trainer(
                sequences=sequences,
                config=config,
                dataset=dataset,
                validation_mode="cv"
            )
            model, cv_metrics = trainer.cross_validate(num_epochs=num_epochs)
            
            # Get F1 score for flat or hierarchical classification
            if 'f1' in cv_metrics['average_metrics']:
                # Regular case - flat classification
                f1_score = cv_metrics['average_metrics']['f1']
            elif 'overall_f1' in cv_metrics['average_metrics']:
                # Hierarchical case with overall F1
                f1_score = cv_metrics['average_metrics']['overall_f1']
            else:
                # Hierarchical case - calculate weighted average
                family_f1 = cv_metrics['average_metrics'].get('family', {}).get('f1', 0)
                genus_f1 = cv_metrics['average_metrics'].get('genus', {}).get('f1', 0) 
                species_f1 = cv_metrics['average_metrics'].get('species', {}).get('f1', 0)
                f1_score = 0.2 * family_f1 + 0.3 * genus_f1 + 0.5 * species_f1
            
            # Update progress bar with current best score
            tune_pbar.set_postfix({
                'best_f1': f"{best_f1:.4f}",
                'curr_f1': f"{f1_score:.4f}"
            })
            # Update progress bar with detailed metrics
            tune_pbar.update(1)
            tune_pbar.set_description(f"{desc} (Best F1: {best_f1:.4f})")
            
            # Save detailed results to file instead of console if not verbose
            trial_results = {
                'params': log_params,
                'f1': f1_score,
                'cv_metrics': cv_metrics
            }
            
            if self.verbose:
                logger.info(f"\nTrial Results:")
                logger.info(f"F1 Score: {f1_score:.4f}")
                logger.info(f"Parameters: {log_params}")
            
            # Track all results
            history.append(trial_results)
            
            improvement = f1_score - best_f1
            if improvement > min_delta:
                best_f1 = f1_score
                best_params = params
                best_cv_metrics = cv_metrics
                best_model = model
                best_visualization_dir = getattr(trainer, 'last_visualization_dir', None)
                # Save the best model checkpoint
                best_model_path = os.path.join("output", "best_tuned_model.pt")
                torch.save(best_model.state_dict(), best_model_path)
                logger.info(f"New best model found! F1 score improvement: {improvement:.4f}")
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                
            # Early stopping check
            if self.mode == 'random' and rounds_without_improvement >= self.early_stopping_rounds:
                logger.info(f"Early stopping triggered after {rounds_without_improvement} rounds without improvement")
                break
        
        # Export detailed results to files instead of console
        top_configs_file, trials_file = self.export_detailed_results(
            history, best_params, best_f1, best_visualization_dir
        )
        
        # Just print a summary to console
        logger.info(f"\nHyperparameter tuning completed.")
        logger.info(f"Best F1 score: {best_f1:.4f}")
        logger.info(f"Top configurations saved to: {top_configs_file}")
        logger.info(f"All trial results saved to: {trials_file}")
        
        tune_pbar.close()
        
        return best_params, best_cv_metrics, best_model, best_visualization_dir
