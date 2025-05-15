import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import os
from typing import Dict, Any, List, Union, Tuple, Optional
from sklearn.metrics import auc
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def export_taxa_classifications(classifications: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Export per-taxa classification results to individual TSV files."""
    os.makedirs(output_dir, exist_ok=True)
    exported_files = []
    
    for level, taxa_data in classifications.items():
        output_path = os.path.join(output_dir, f"{level}_classifications.tsv")
        rows = []
        
        for taxa_name, stats in taxa_data.items():
            misclassified_to = stats.get("misclassified_to", {})
            misclassifications = "; ".join([f"{wrong_label} ({count})" 
                                          for wrong_label, count in misclassified_to.items()])
            
            total = stats["total"]
            correct = stats["correct"]
            false = total - correct
            accuracy = correct / total if total > 0 else 0
            
            # Calculate percentages
            seen_total = stats.get("seen_total", 0)
            unseen_total = stats.get("unseen_total", 0)
            low_confidence = stats.get("low_confidence", 0)
            unseen_correct = stats.get("unseen_correct_unknown", 0)
            unseen_incorrect = stats.get("unseen_incorrect", 0)
            unseen_low_conf = stats.get("unseen_low_confidence", 0)
            
            row = {
                "Taxa": taxa_name,
                "Total Sequences": total,
                "Correct Classifications": correct,
                "Incorrect Classifications": false,
                "Accuracy": f"{accuracy:.4f}",
                "Is Unseen": "Yes" if stats.get("is_unseen", False) else "No",
                "Total Seen": seen_total,
                "Total Unseen": unseen_total,
                "Low Confidence Count": low_confidence,
                "Low Confidence Rate": f"{low_confidence/total:.4f}" if total > 0 else "0",
                "Unseen Correct Unknown": unseen_correct,
                "Unseen Incorrect": unseen_incorrect,
                "Unseen Low Confidence": unseen_low_conf,
                "Misclassified To": misclassifications if misclassifications else "N/A"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(["Accuracy", "Total Sequences"], ascending=[False, False])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            headers = [
                "Taxa", "Total Sequences", "Correct Classifications", "Incorrect Classifications", 
                "Accuracy", "Is Unseen", "Total Seen", "Total Unseen", "Low Confidence Count",
                "Low Confidence Rate", "Unseen Correct Unknown", "Unseen Incorrect", 
                "Unseen Low Confidence", "Misclassified To"
            ]
            f.write("\t".join(headers) + "\n")
            
            for _, row in df.iterrows():
                values = [
                    str(row["Taxa"]),
                    str(row["Total Sequences"]),
                    str(row["Correct Classifications"]),
                    str(row["Incorrect Classifications"]),
                    str(row["Accuracy"]),
                    str(row["Is Unseen"]),
                    str(row["Total Seen"]),
                    str(row["Total Unseen"]),
                    str(row["Low Confidence Count"]),
                    str(row["Low Confidence Rate"]),
                    str(row["Unseen Correct Unknown"]),
                    str(row["Unseen Incorrect"]),
                    str(row["Unseen Low Confidence"]),
                    str(row["Misclassified To"])
                ]
                f.write("\t".join(values) + "\n")
        
            exported_files.append(output_path)
    
    if exported_files:
        logger.info(f"Exported classifications to {len(exported_files)} files in {output_dir}")

def export_cv_metrics_to_tsv(cv_metrics: Dict, output_path: str) -> None:
    """Export detailed cross-validation metrics to a TSV file."""
    if not cv_metrics.get('average_metrics') or not cv_metrics.get('std_dev') or not cv_metrics.get('fold_metrics'):
        logger.warning("No metrics to export")
        return
    
    num_folds = len(cv_metrics['fold_metrics'])
    rows = []
    
    for level in cv_metrics['average_metrics'].keys():
        for fold_idx in range(num_folds):
            metrics = cv_metrics['fold_metrics'][fold_idx]['metrics'][level]
            
            # Get classification data
            total = sum(len(taxa["confidences"]) for taxa in metrics["classifications"].values())
            seen_total = sum(taxa["seen_total"] for taxa in metrics["classifications"].values())
            unseen_total = sum(taxa["unseen_total"] for taxa in metrics["classifications"].values())
            
            # Get error counts
            seen_correct = sum(1 for taxa in metrics["classifications"].values() 
                             if not taxa["is_unseen"] and taxa["correct"] > 0)
            seen_incorrect = sum(len(taxa["misclassified_to"]) for taxa in metrics["classifications"].values() 
                               if not taxa["is_unseen"])
            
            # Get confidence counts
            low_conf_total = sum(taxa["low_confidence"] for taxa in metrics["classifications"].values())
            seen_low_conf = sum(taxa["low_confidence"] for taxa in metrics["classifications"].values() 
                              if not taxa["is_unseen"])
            unseen_low_conf = sum(taxa["unseen_low_confidence"] for taxa in metrics["classifications"].values())
            
            # Get unseen performance
            unseen_correct = sum(taxa["unseen_correct_unknown"] for taxa in metrics["classifications"].values())
            unseen_incorrect = sum(taxa["unseen_incorrect"] for taxa in metrics["classifications"].values())
            
            row = {
                "Level": level.capitalize(),
                "Fold": f"{fold_idx + 1}",
                "Threshold": f"{metrics.get('threshold', 'N/A')}",
                "Total Sequences": total,
                "Seen Total": seen_total,
                "Seen Correct": seen_correct,
                "Seen Incorrect": seen_incorrect,
                "Seen Low Confidence": seen_low_conf,
                "Unseen Total": unseen_total,
                "Unseen Correct": unseen_correct,
                "Unseen Incorrect": unseen_incorrect,
                "Unseen Low Confidence": unseen_low_conf,
                "Total Low Confidence": low_conf_total,
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}"
            }
            
            rows.append(row)
        
        # Calculate averages
        fold_data = [fold['metrics'][level]["classifications"] for fold in cv_metrics['fold_metrics']]
        # Calculate means and standard deviations
        def get_stats(fn):
            values = [fn(fold.values()) for fold in fold_data]
            return np.mean(values), np.std(values)

        avg_total, std_total = get_stats(lambda x: sum(len(taxa["confidences"]) for taxa in x))
        avg_seen, std_seen = get_stats(lambda x: sum(taxa["seen_total"] for taxa in x))
        avg_seen_correct, std_seen_correct = get_stats(lambda x: sum(1 for taxa in x if not taxa["is_unseen"] and taxa["correct"] > 0))
        avg_seen_incorrect, std_seen_incorrect = get_stats(lambda x: sum(len(taxa["misclassified_to"]) for taxa in x if not taxa["is_unseen"]))
        avg_seen_low_conf, std_seen_low_conf = get_stats(lambda x: sum(taxa["low_confidence"] for taxa in x if not taxa["is_unseen"]))
        avg_unseen, std_unseen = get_stats(lambda x: sum(taxa["unseen_total"] for taxa in x))
        avg_unseen_correct, std_unseen_correct = get_stats(lambda x: sum(taxa["unseen_correct_unknown"] for taxa in x))
        avg_unseen_incorrect, std_unseen_incorrect = get_stats(lambda x: sum(taxa["unseen_incorrect"] for taxa in x))
        avg_unseen_low_conf, std_unseen_low_conf = get_stats(lambda x: sum(taxa["unseen_low_confidence"] for taxa in x))
        avg_low_conf, std_low_conf = get_stats(lambda x: sum(taxa["low_confidence"] for taxa in x))
        
        avg = cv_metrics['average_metrics'][level]
        std = cv_metrics['std_dev'][level]
        row = {
            "Level": level.capitalize(),
            "Fold": "Average",
            "Threshold": f"{cv_metrics['fold_metrics'][0]['metrics'][level].get('threshold', 'N/A')}",
            "Total Sequences": f"{avg_total:.1f} (±{std_total:.1f})",
            "Seen Total": f"{avg_seen:.1f} (±{std_seen:.1f})",
            "Seen Correct": f"{avg_seen_correct:.1f} (±{std_seen_correct:.1f})",
            "Seen Incorrect": f"{avg_seen_incorrect:.1f} (±{std_seen_incorrect:.1f})",
            "Seen Low Confidence": f"{avg_seen_low_conf:.1f} (±{std_seen_low_conf:.1f})",
            "Unseen Total": f"{avg_unseen:.1f} (±{std_unseen:.1f})",
            "Unseen Correct": f"{avg_unseen_correct:.1f} (±{std_unseen_correct:.1f})",
            "Unseen Incorrect": f"{avg_unseen_incorrect:.1f} (±{std_unseen_incorrect:.1f})",
            "Unseen Low Confidence": f"{avg_unseen_low_conf:.1f} (±{std_unseen_low_conf:.1f})",
            "Total Low Confidence": f"{avg_low_conf:.1f} (±{std_low_conf:.1f})",
            "Precision": f"{avg['precision']:.4f} (±{std['precision']:.4f})",
            "Recall": f"{avg['recall']:.4f} (±{std['recall']:.4f})",
            "F1": f"{avg['f1']:.4f} (±{std['f1']:.4f})"
        }
        rows.append(row)
        rows.append({
            "Level": "", "Fold": "", "Threshold": "",
            "Total Sequences": "", "Seen Total": "", "Seen Correct": "",
            "Seen Incorrect": "", "Seen Low Confidence": "", "Unseen Total": "",
            "Unseen Correct": "", "Unseen Incorrect": "", "Unseen Low Confidence": "",
            "Total Low Confidence": "", "Precision": "", "Recall": "", "F1": ""
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)

def plot_loss_curves(train_losses: np.ndarray, val_losses: np.ndarray, output_path: str) -> None:
    """Plot training and validation loss curves with running averages for smoothing."""
    epochs = list(range(1, len(train_losses) + 1))
    
    # Set better style and color palette
    sns.set(style="whitegrid", palette="deep")
    plt.figure(figsize=(12, 7))
    
    # Define better colors for the plot
    train_color = '#FF5733'  # Bright orange/red
    val_color = '#3498DB'    # Bright blue
    
    # Plot raw data with nicer markers
    plt.plot(epochs, train_losses, 'o-', color=train_color, alpha=1.0, 
             markersize=12, label='Training Loss', markerfacecolor=train_color, 
             markeredgecolor='white', markeredgewidth=0.5, linewidth=5)
    plt.plot(epochs, val_losses, 's-', color=val_color, alpha=1.0,
             markersize=12, label='Validation Loss', markerfacecolor=val_color, 
             markeredgecolor='white', markeredgewidth=0.5, linewidth=5)
    
    # Apply smoothing with better visualization
    window = min(5, len(train_losses) // 10)
    if window > 1:
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        smooth_epochs = epochs[window-1:]
        
        plt.plot(smooth_epochs, train_smooth, '-', color=train_color, label='Training Loss (Smoothed)', 
                linewidth=2.5)
        plt.plot(smooth_epochs, val_smooth, '-', color=val_color, label='Validation Loss (Smoothed)', 
                linewidth=2.5)
    
    # Enhance the plot appearance
    plt.title("Training and Validation Loss Curves", fontsize=16, pad=15)
    plt.xlabel("Epoch", fontsize=12, labelpad=10)
    plt.ylabel("Loss", fontsize=12, labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=False, fontsize=10, facecolor='white', edgecolor='gray', framealpha=0.9)
    
    # Set y-axis to start from zero for better perspective
    plt.ylim(bottom=0)
    
    # Add subtle background shading to highlight the curves
    plt.gca().set_facecolor('#f9f9f9')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()

def plot_roc_curves(threshold_metrics: Dict[str, Dict], output_path: str, level: Optional[str] = None) -> None:
    """Plot ROC-like curves showing true positive rate vs false positive rate across confidence thresholds."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    thresholds = sorted([float(t) for t in threshold_metrics.keys()])
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        metrics = threshold_metrics[str(threshold)]
        tpr.append(metrics['sensitivity'])
        fpr.append(1 - metrics['specificity'])
    
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    
    for i, threshold in enumerate(thresholds[::10]):
        plt.annotate(f'{threshold:.2f}', 
                    (fpr[i*10], tpr[i*10]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    title = 'ROC Curve - Confidence Threshold Analysis'
    if level:
        title += f' for {level.capitalize()}'
    plt.title(title)
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()

def plot_confidence_metrics(threshold_metrics: Dict[str, Dict], output_path: str, level: Optional[str] = None) -> None:
    """Plot sensitivity and specificity against confidence thresholds."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    thresholds = sorted([float(t) for t in threshold_metrics.keys()])
    sensitivity = [threshold_metrics[str(t)]['sensitivity'] for t in thresholds]
    specificity = [threshold_metrics[str(t)]['specificity'] for t in thresholds]
    
    plt.plot(thresholds, sensitivity, 'b-', label='Sensitivity')
    plt.plot(thresholds, specificity, 'r-', label='Specificity')
    
    optimal_idx = np.argmax(np.array(sensitivity) + np.array(specificity))
    optimal_threshold = thresholds[optimal_idx]
    
    plt.axvline(x=optimal_threshold, color='g', linestyle='--', alpha=0.5,
                label=f'Optimal threshold = {optimal_threshold:.2f}')
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    title = 'Confidence Threshold Analysis'
    if level:
        title += f' for {level.capitalize()}'
    plt.title(title)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()

def export_confidence_metrics_to_tsv(threshold_metrics: Dict[str, Dict], output_path: str) -> None:
    """Export confidence threshold analysis metrics to TSV file."""
    rows = []
    
    thresholds = sorted([float(t) for t in threshold_metrics.keys()])
    for threshold in thresholds:
        metrics = threshold_metrics[str(threshold)]
        
        row = {
            "Threshold": f"{threshold:.2f}",
            "Sensitivity": f"{metrics['sensitivity']:.4f}",
            "Specificity": f"{metrics['specificity']:.4f}",
            "True Positives": sum(metrics['true_positives'].values()),
            "False Positives": sum(metrics['false_positives'].values()),
            "True Negatives": sum(metrics['true_negatives'].values()),
            "False Negatives": sum(metrics['false_negatives'].values()),
            "Unseen Taxa": len(metrics['unseen_taxa'])
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)

def plot_detailed_metrics(cv_metrics: Dict, output_path: str) -> None:
    """Create a single plot showing detailed classification performance by taxonomic rank."""
    # Set better style
    sns.set_style("white")
    plt.rcParams.update({'font.size': 12})
    
    # Create a single figure
    plt.figure(figsize=(14, 8))
    
    # Define colors for different categories
    colors = {
        'seen_correct': '#2ecc71',      # Green
        'seen_incorrect': '#e74c3c',    # Red
        'unseen_correct': '#3498db',    # Blue
        'unseen_incorrect': '#e67e22'   # Orange
    }
    
    # Get taxonomic ranks and prepare data
    taxonomic_ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    ranks_present = [rank for rank in taxonomic_ranks if rank in cv_metrics['average_metrics']]
    x = np.arange(len(ranks_present))
    width = 0.2  # Width of each bar
    
    # Prepare statistics for each rank
    stats = []
    for rank in ranks_present:
        fold_data = [fold['metrics'][rank]["classifications"] for fold in cv_metrics['fold_metrics']]
        
        def get_percentages(fn):
            values = [fn(fold.values()) for fold in fold_data]
            return np.mean(values), np.std(values)
        
        # Calculate statistics
        seen_total, _ = get_percentages(lambda x: sum(taxa["seen_total"] for taxa in x))
        seen_correct, seen_correct_std = get_percentages(lambda x: sum(1 for taxa in x if not taxa["is_unseen"] and taxa["correct"] > 0))
        seen_incorrect, seen_incorrect_std = get_percentages(lambda x: sum(len(taxa["misclassified_to"]) for taxa in x if not taxa["is_unseen"]))
        
        unseen_total, _ = get_percentages(lambda x: sum(taxa["unseen_total"] for taxa in x))
        unseen_correct, unseen_correct_std = get_percentages(lambda x: sum(taxa["unseen_correct_unknown"] for taxa in x))
        unseen_incorrect, unseen_incorrect_std = get_percentages(lambda x: sum(taxa["unseen_incorrect"] for taxa in x))
        
        # Convert to percentages
        stats.append({
            'rank': rank,
            'seen_correct_pct': (seen_correct / seen_total * 100 if seen_total > 0 else 0),
            'seen_correct_std': (seen_correct_std / seen_total * 100 if seen_total > 0 else 0),
            'seen_incorrect_pct': (seen_incorrect / seen_total * 100 if seen_total > 0 else 0),
            'seen_incorrect_std': (seen_incorrect_std / seen_total * 100 if seen_total > 0 else 0),
            'unseen_correct_pct': (unseen_correct / unseen_total * 100 if unseen_total > 0 else 0),
            'unseen_correct_std': (unseen_correct_std / unseen_total * 100 if unseen_total > 0 else 0),
            'unseen_incorrect_pct': (unseen_incorrect / unseen_total * 100 if unseen_total > 0 else 0),
            'unseen_incorrect_std': (unseen_incorrect_std / unseen_total * 100 if unseen_total > 0 else 0),
        })
    
    # Plot bars
    seen_correct = plt.bar(x - width*1.5, [s['seen_correct_pct'] for s in stats], width, 
                          label='Seen - Correct', color=colors['seen_correct'])
    seen_incorrect = plt.bar(x - width/2, [s['seen_incorrect_pct'] for s in stats], width,
                            label='Seen - Incorrect', color=colors['seen_incorrect'])
    unseen_correct = plt.bar(x + width/2, [s['unseen_correct_pct'] for s in stats], width,
                            label='Unseen - Correct', color=colors['unseen_correct'])
    unseen_incorrect = plt.bar(x + width*1.5, [s['unseen_incorrect_pct'] for s in stats], width,
                              label='Unseen - Incorrect', color=colors['unseen_incorrect'])
    
    # Add error bars
    plt.errorbar(x - width*1.5, [s['seen_correct_pct'] for s in stats],
                yerr=[s['seen_correct_std'] for s in stats], fmt='none', color='black', capsize=5)
    plt.errorbar(x - width/2, [s['seen_incorrect_pct'] for s in stats],
                yerr=[s['seen_incorrect_std'] for s in stats], fmt='none', color='black', capsize=5)
    plt.errorbar(x + width/2, [s['unseen_correct_pct'] for s in stats],
                yerr=[s['unseen_correct_std'] for s in stats], fmt='none', color='black', capsize=5)
    plt.errorbar(x + width*1.5, [s['unseen_incorrect_pct'] for s in stats],
                yerr=[s['unseen_incorrect_std'] for s in stats], fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.xlabel('Taxonomic Rank', fontsize=14, labelpad=10)
    plt.ylabel('Percentage (%)', fontsize=14, labelpad=10)
    plt.title("Classification Performance by Taxonomic Rank", fontsize=16, pad=20)
    plt.xticks(x, [r.capitalize() for r in ranks_present], rotation=45)
    plt.ylim(0, 110)  # More space for error bars
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Place legend in a good position with better styling
    plt.legend(loc='upper right', framealpha=0.95, frameon=True, 
               facecolor='white', edgecolor='gray', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_metrics(metrics: Dict, output_path: str) -> None:
    """Plot performance metrics with confidence intervals."""
    expected_levels = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    metrics_list = ["precision", "recall", "f1"]
    
    is_cv = "average_metrics" in metrics
    metrics_key = "average_metrics" if is_cv else "mean_metrics"
    error_key = "std_dev" if is_cv else "confidence_intervals"

    taxonomic_levels = [level for level in expected_levels if level in metrics[metrics_key]]
    
    if not taxonomic_levels or not metrics[metrics_key]:
        logger.warning("No metrics data found for combined metrics plot")
        return
        
    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot metrics
    records = []
    for level in taxonomic_levels:
        mean_dict = metrics[metrics_key][level]
        error_dict = metrics[error_key][level]
        for metric in metrics_list:
            if metric not in mean_dict:
                continue
                
            mean_value = mean_dict[metric]
            if is_cv:
                std = error_dict[metric]
                ci_lower = max(0, mean_value - std)  # Ensure non-negative values
                ci_upper = min(1, mean_value + std)  # Cap at 1.0
            else:
                ci_lower = error_dict[metric][0]
                ci_upper = error_dict[metric][1]
                
            records.append({
                "level": level.capitalize(),
                "metric": metric.capitalize(),
                "mean": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            })
    
    if not records:
        logger.warning("No metric records could be generated for combined metrics plot")
        return
    
    df = pd.DataFrame(records)
    df["level"] = pd.Categorical(df["level"], 
                               categories=[lvl.capitalize() for lvl in taxonomic_levels], 
                               ordered=True)
    
    x = np.arange(len(taxonomic_levels))
    width = 0.2
    metric_names = sorted(list(set(r["metric"] for r in records)))
    colors = sns.color_palette("Set2", len(metric_names))
    hatches = ['/', 'o', 'x', '\\']
    alpha = 0.8
    
    # Plot a bar for each metric
    for i, metric in enumerate(metric_names):
        metric_df = df[df["metric"] == metric].sort_values("level")
        if metric_df.empty:
            continue
            
        means = metric_df["mean"].values
        ci_lower = metric_df["ci_lower"].values
        ci_upper = metric_df["ci_upper"].values
        
        # Calculate error bar heights
        yerr = np.array([means - ci_lower, ci_upper - means])
        
        # Calculate bar positions
        positions = x + (i * width) - ((len(metric_names) - 1) * width / 2)
        
        # Plot bars with error bars
        bars = ax.bar(positions, means, width, color=colors[i], capsize=4,
               label=metric, hatch=hatches[i % len(hatches)], 
               yerr=yerr, alpha=alpha)
               
        # Add value labels on top of each bar
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                height + 0.02, 
                f'{means[j]:.2f}', 
                ha='center', 
                va='bottom',
                fontsize=8)
    
    # Set up the plot
    ax.set_xticks(x)
    ax.set_xticklabels([lvl.capitalize() for lvl in taxonomic_levels], rotation=45)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel("Score")
    ax.set_title("Performance Metrics by Taxonomic Rank")
    ax.set_ylim(0, 1.1)  # Set y-axis limits to ensure visibility
    ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid for readability
    ax.legend(title="Metric", fontsize="small", loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_metrics_on_axis(ax, metrics, taxonomic_levels, metrics_list, metrics_key, error_key, is_cv, title, metric_display_names=None):
    """Helper function to plot metrics on a given axis."""
    if metric_display_names is None:
        metric_display_names = {m: m.capitalize() for m in metrics_list}
    
    records = []
    for level in taxonomic_levels:
        mean_dict = metrics[metrics_key][level]
        error_dict = metrics[error_key][level]
        for metric in metrics_list:
            if metric not in mean_dict:
                continue
                
            mean_value = mean_dict[metric]
            if is_cv:
                std = error_dict[metric]
                ci_lower = mean_value - std
                ci_upper = mean_value + std
            else:
                ci_lower = error_dict[metric][0]
                ci_upper = error_dict[metric][1]
                
            records.append({
                "level": level.capitalize(),
                "metric": metric_display_names.get(metric, metric.capitalize()),
                "mean": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            })
    
    df = pd.DataFrame(records)
    df["level"] = pd.Categorical(df["level"], 
                               categories=[lvl.capitalize() for lvl in taxonomic_levels], 
                               ordered=True)
    
    x = np.arange(len(taxonomic_levels))
    width = 0.2
    metric_names = [metric_display_names.get(m, m.capitalize()) for m in metrics_list if m in df["metric"].unique()]
    colors = sns.color_palette("Set2", len(metric_names))
    hatches = ['/', 'o', 'x', '\\']
    alpha = 0.8
    
    for i, metric in enumerate(metric_names):
        metric_df = df[df["metric"] == metric].set_index("level")
        means = metric_df["mean"]
        ci_lower = metric_df["ci_lower"]
        ci_upper = metric_df["ci_upper"]
        positions = x + i * width - (width * len(metric_names)) / 2
        
        yerr = np.array([means - ci_lower, ci_upper - means])
        bars = ax.bar(positions, means, width, color=colors[i], capsize=4,
               label=metric, hatch=hatches[i], yerr=yerr,
               alpha=alpha)
               
        # Add value labels on top of each bar
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                height + 0.02, 
                f'{means.iloc[j]:.2f}', 
                ha='center', 
                va='bottom',
                fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([lvl.capitalize() for lvl in taxonomic_levels], rotation=45)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(title="Metric", fontsize="small", loc='lower right')

def plot_threshold_curves(threshold_curves: Dict[str, Dict[str, List[float]]], output_path: str) -> None:
    """Plot threshold optimization curves for each taxonomic level.
    
    Args:
        threshold_curves: Dictionary mapping levels to performance curves
        output_path: Path to save the output file
    """
    levels = list(threshold_curves.keys())
    n_levels = len(levels)
    
    # Set up a grid of subplots - one for each taxonomic level
    rows = (n_levels + 1) // 2  # Ceiling division to get enough rows
    cols = 2
    
    # Create figure with better size and style
    plt.figure(figsize=(14, 4 * rows))
    sns.set_style("whitegrid")
    
    # Plot performance curves for each level
    for i, level in enumerate(levels):
        ax = plt.subplot(rows, cols, i + 1)
        
        curves = threshold_curves[level]
        thresholds = curves["thresholds"]
        precision = curves["precision"]
        recall = curves["recall"]
        f1 = curves["f1"]
        
        # Find the best F1 score and corresponding threshold
        best_f1_idx = np.argmax(f1)
        best_f1 = f1[best_f1_idx]
        best_f1_threshold = thresholds[best_f1_idx]
        
        # Plot the curves
        plt.plot(thresholds, precision, 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recall, 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1, 'g-', label='F1', linewidth=2)
        
        # Mark the best F1 score
        plt.plot(best_f1_threshold, best_f1, 'o', 
                markersize=8, markerfacecolor='gold', markeredgecolor='black',
                label=f'Best F1 ({best_f1:.3f} at {best_f1_threshold:.2f})')
        
        # Add a vertical line at the best threshold
        plt.axvline(x=best_f1_threshold, color='gray', linestyle='--', alpha=0.5)
        
        # Set axis labels and title
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score')
        plt.title(f'{level.capitalize()} - Threshold Optimization')
        
        # Set y-axis limits
        plt.ylim([0, 1.05])
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def export_threshold_optimization_to_tsv(optimized_thresholds: List[Dict[str, float]], 
                                        output_path: str) -> None:
    """Export threshold optimization results to TSV file.
    
    Args:
        optimized_thresholds: List of optimized thresholds for each fold
        output_path: Path to save the output file
    """
    if not optimized_thresholds:
        return
    
    import pandas as pd
    
    # Get all levels
    levels = set()
    for fold_thresholds in optimized_thresholds:
        levels.update(fold_thresholds.keys())
    levels = sorted(levels)
    
    # Prepare rows for each fold
    rows = []
    for fold_idx, fold_thresholds in enumerate(optimized_thresholds):
        row = {"Fold": f"Fold {fold_idx + 1}"}
        for level in levels:
            if level in fold_thresholds:
                row[level.capitalize()] = f"{fold_thresholds[level]:.4f}"
            else:
                row[level.capitalize()] = "N/A"
        rows.append(row)
    
    # Calculate average thresholds
    avg_row = {"Fold": "Average"}
    std_row = {"Fold": "Std. Dev."}
    
    for level in levels:
        fold_values = [fold_thresholds.get(level, np.nan) for fold_thresholds in optimized_thresholds]
        fold_values = [v for v in fold_values if not np.isnan(v)]
        
        if fold_values:
            avg = np.mean(fold_values)
            std = np.std(fold_values)
            avg_row[level.capitalize()] = f"{avg:.4f}"
            std_row[level.capitalize()] = f"{std:.4f}"
        else:
            avg_row[level.capitalize()] = "N/A"
            std_row[level.capitalize()] = "N/A"
    
    rows.append(avg_row)
    rows.append(std_row)
    
    # Create and save DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
