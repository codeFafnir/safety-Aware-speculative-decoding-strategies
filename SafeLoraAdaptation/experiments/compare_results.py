#!/usr/bin/env python
"""
Compare baseline and SafeLoRA results.

This script:
1. Loads baseline and SafeLoRA evaluation results
2. Computes improvement metrics
3. Generates comparison tables and visualizations

Usage:
    # Option 1: Run with default config (modify COMPARISON_CONFIG in src/experiment_config.py)
    python experiments/compare_results.py
    
    # Option 2: Import and customize in Python
    from experiments.compare_results import run_comparison
    from src.experiment_config import ComparisonConfig
    
    config = ComparisonConfig(
        baseline_results=["results/baseline/*.json"],
        safelora_results=["results/safelora/*.json"],
        output_dir="results/my_comparison",
    )
    run_comparison(config)
"""

import os
import sys
import json
import glob
from typing import Dict, List, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_config import COMPARISON_CONFIG, ComparisonConfig


def load_results(patterns: List[str]) -> List[Dict]:
    """Load results from JSON files matching patterns."""
    results = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, 'r') as f:
                data = json.load(f)
                data['_source_file'] = path
                results.append(data)
    return results


def compute_comparison(baseline_results: List[Dict], safelora_results: List[Dict]) -> pd.DataFrame:
    """Compute comparison metrics between baseline and SafeLoRA."""
    
    rows = []
    
    for baseline, safelora in zip(baseline_results, safelora_results):
        # Extract config info
        config_info = {
            "model": baseline.get("config", {}).get("base_model", "unknown"),
            "quantization": "4-bit" if baseline.get("config", {}).get("load_in_4bit") else 
                           ("8-bit" if baseline.get("config", {}).get("load_in_8bit") else "FP16"),
            "num_samples": baseline.get("config", {}).get("num_samples", 0),
        }
        
        # SafeLoRA specific info
        safelora_config = {
            "layers_projected": safelora.get("projection_stats", {}).get("num_projected", 0),
            "total_layers": safelora.get("projection_stats", {}).get("total_layers", 0),
            "mean_pdst": safelora.get("projection_stats", {}).get("mean_pdst", 0),
        }
        
        # Compare each metric
        baseline_metrics = baseline.get("results", {})
        safelora_metrics = safelora.get("results", {})
        
        for metric_name in baseline_metrics:
            if metric_name in safelora_metrics:
                baseline_score = baseline_metrics[metric_name].get("score", 0)
                safelora_score = safelora_metrics[metric_name].get("score", 0)
                
                improvement = safelora_score - baseline_score
                improvement_pct = (improvement / max(baseline_score, 1e-6)) * 100
                
                row = {
                    **config_info,
                    **safelora_config,
                    "metric": metric_name,
                    "baseline_score": baseline_score,
                    "safelora_score": safelora_score,
                    "improvement": improvement,
                    "improvement_pct": improvement_pct,
                }
                rows.append(row)
    
    return pd.DataFrame(rows)


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comparison visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 1. Bar chart comparing baseline vs SafeLoRA scores
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = df['metric'].unique()
    x = range(len(metrics))
    width = 0.35
    
    baseline_scores = [df[df['metric'] == m]['baseline_score'].mean() for m in metrics]
    safelora_scores = [df[df['metric'] == m]['safelora_score'].mean() for m in metrics]
    
    bars1 = ax.bar([i - width/2 for i in x], baseline_scores, width, label='Baseline', color='#ff6b6b')
    bars2 = ax.bar([i + width/2 for i in x], safelora_scores, width, label='SafeLoRA', color='#4ecdc4')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Safety Scores: Baseline vs SafeLoRA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'safety_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Improvement chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = [df[df['metric'] == m]['improvement'].mean() for m in metrics]
    colors = ['#4ecdc4' if imp >= 0 else '#ff6b6b' for imp in improvements]
    
    bars = ax.barh(metrics, improvements, color=colors)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Improvement (SafeLoRA - Baseline)', fontsize=12)
    ax.set_title('Safety Improvement with SafeLoRA', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax.annotate(f'{imp:+.3f}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if width >= 0 else -5, 0),
                    textcoords="offset points",
                    ha='left' if width >= 0 else 'right',
                    va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_chart.png'), dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def print_summary(df: pd.DataFrame):
    """Print comparison summary."""
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: BASELINE vs SAFELORA")
    print("=" * 80)
    
    # Per-metric summary
    for metric in df['metric'].unique():
        metric_df = df[df['metric'] == metric]
        
        baseline_avg = metric_df['baseline_score'].mean()
        safelora_avg = metric_df['safelora_score'].mean()
        improvement_avg = metric_df['improvement'].mean()
        improvement_pct_avg = metric_df['improvement_pct'].mean()
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Baseline:    {baseline_avg:.4f}")
        print(f"  SafeLoRA:    {safelora_avg:.4f}")
        print(f"  Improvement: {improvement_avg:+.4f} ({improvement_pct_avg:+.1f}%)")
    
    # Overall summary
    print("\n" + "-" * 80)
    print("OVERALL:")
    overall_baseline = df.groupby('metric')['baseline_score'].mean().mean()
    overall_safelora = df.groupby('metric')['safelora_score'].mean().mean()
    overall_improvement = overall_safelora - overall_baseline
    
    print(f"  Average Baseline Score:  {overall_baseline:.4f}")
    print(f"  Average SafeLoRA Score:  {overall_safelora:.4f}")
    print(f"  Average Improvement:     {overall_improvement:+.4f}")
    
    # Projection stats
    if 'layers_projected' in df.columns:
        print("\nProjection Statistics:")
        print(f"  Layers Projected: {df['layers_projected'].iloc[0]}/{df['total_layers'].iloc[0]}")
        print(f"  Mean Pdst: {df['mean_pdst'].iloc[0]:.4f}")
    
    print("=" * 80)


def run_comparison(config: ComparisonConfig = None):
    """
    Run comparison between baseline and SafeLoRA results.
    
    Args:
        config: Comparison configuration. If None, uses COMPARISON_CONFIG from experiment_config.py
    """
    if config is None:
        config = COMPARISON_CONFIG
    
    # Load results
    print("Loading baseline results...")
    baseline_results = load_results(config.baseline_results)
    print(f"  Found {len(baseline_results)} baseline result files")
    
    print("Loading SafeLoRA results...")
    safelora_results = load_results(config.safelora_results)
    print(f"  Found {len(safelora_results)} SafeLoRA result files")
    
    if not baseline_results or not safelora_results:
        print("Error: No result files found!")
        return None
    
    # Compute comparison
    df = compute_comparison(baseline_results, safelora_results)
    
    # Print summary
    print_summary(df)
    
    # Create visualizations
    create_visualizations(df, config.output_dir)
    
    # Save comparison data
    os.makedirs(config.output_dir, exist_ok=True)
    csv_path = os.path.join(config.output_dir, 'comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nComparison data saved to {csv_path}")
    
    return df


def main():
    """Run comparison with default configuration."""
    run_comparison(COMPARISON_CONFIG)


if __name__ == "__main__":
    main()
