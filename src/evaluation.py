"""
Model Evaluation and Visualization Module

This module provides comprehensive evaluation metrics and visualizations
for IoT device classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Any
import os


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, label_mapping: Dict[str, int], 
                 output_dir: str = 'results/figures'):
        """
        Initialize the evaluator.
        
        Args:
            label_mapping: Mapping from label names to indices
            output_dir: Directory for saving figures
        """
        self.label_mapping = label_mapping
        self.class_names = list(label_mapping.keys())
        self.n_classes = len(label_mapping)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = 'Model') -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        metrics['precision_per_class'] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics['recall_per_class'] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics['f1_per_class'] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, zero_division=0
        )
        
        return metrics
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models and create summary DataFrame.
        
        Args:
            results: Dictionary of evaluation results for each model
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1 (Macro)': metrics['f1_macro'],
                'F1 (Weighted)': metrics['f1_weighted']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1 (Macro)', ascending=False)
        
        return df
    
    def print_evaluation(self, metrics: Dict[str, Any]):
        """
        Print formatted evaluation results.
        
        Args:
            metrics: Evaluation metrics dictionary
        """
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS: {metrics['model_name']}")
        print("="*60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
        print(f"  Precision (Weight): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
        print(f"  Recall (Weight):    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Weight):  {metrics['f1_weighted']:.4f}")
        print(f"\nClassification Report:")
        print(metrics['classification_report'])


class Visualizer:
    """Visualization utilities for model evaluation."""
    
    def __init__(self, label_mapping: Dict[str, int],
                 output_dir: str = 'results/figures'):
        """
        Initialize the visualizer.
        
        Args:
            label_mapping: Mapping from label names to indices
            output_dir: Directory for saving figures
        """
        self.label_mapping = label_mapping
        self.class_names = list(label_mapping.keys())
        self.n_classes = len(label_mapping)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str,
                              normalize: bool = True, save: bool = True) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            normalize: Whether to normalize the matrix
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            cm_display = cm
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        fig_size = max(8, self.n_classes * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save:
            filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                              save: bool = True) -> plt.Figure:
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i * width, comparison_df[metric], width, 
                         label=metric, alpha=0.8)
            for bar, val in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                                model_name: str, top_n: int = 20,
                                save: bool = True) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            importance_dict: Dictionary of feature importances
            model_name: Name of the model
            top_n: Number of top features to show
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        items = list(importance_dict.items())[:top_n]
        features, importances = zip(*items)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        
        bars = ax.barh(y_pos, importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}', 
                    fontsize=14, fontweight='bold')
        
        for bar, imp in zip(bars, importances):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{imp:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_class_distribution(self, y: np.ndarray, title: str = 'Class Distribution',
                                save: bool = True) -> plt.Figure:
        """
        Plot class distribution.
        
        Args:
            y: Labels array
            title: Plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        unique, counts = np.unique(y, return_counts=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))
        bars = ax1.bar(range(len(unique)), counts, color=colors)
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'{title} (Bar Chart)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(unique)))
        
        if len(self.class_names) <= len(unique):
            ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        ax2.pie(counts, labels=self.class_names if len(self.class_names) <= len(unique) else unique,
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title(f'{title} (Pie Chart)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filename = f'{title.lower().replace(" ", "_")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_per_class_metrics(self, metrics: Dict[str, Any], model_name: str,
                               save: bool = True) -> plt.Figure:
        """
        Plot per-class precision, recall, and F1 scores.
        
        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(self.n_classes)
        width = 0.25
        
        ax.bar(x - width, metrics['precision_per_class'], width, 
               label='Precision', alpha=0.8, color='#2ecc71')
        ax.bar(x, metrics['recall_per_class'], width, 
               label='Recall', alpha=0.8, color='#3498db')
        ax.bar(x + width, metrics['f1_per_class'], width, 
               label='F1-Score', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Per-Class Metrics - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f'per_class_metrics_{model_name.lower().replace(" ", "_")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_training_summary(self, results: Dict[str, Dict],
                              save: bool = True) -> plt.Figure:
        """
        Create a comprehensive training summary plot.
        
        Args:
            results: Dictionary of results for all models
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        
        ax1 = fig.add_subplot(2, 2, 1)
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax1.bar(models, accuracies, color=colors)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax2 = fig.add_subplot(2, 2, 2)
        f1_scores = [results[m]['f1_macro'] for m in models]
        bars = ax2.bar(models, f1_scores, color=colors)
        ax2.set_ylabel('F1-Score (Macro)', fontsize=12)
        ax2.set_title('Model F1-Score Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        for bar, f1 in zip(bars, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax3 = fig.add_subplot(2, 2, 3)
        best_model = max(results.keys(), key=lambda x: results[x]['f1_macro'])
        cm = results[best_model]['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax3)
        ax3.set_title(f'Best Model ({best_model}) Confusion Matrix', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        
        ax4 = fig.add_subplot(2, 2, 4)
        metrics_data = {
            'Accuracy': accuracies,
            'F1 (Macro)': f1_scores,
            'Precision': [results[m]['precision_macro'] for m in models],
            'Recall': [results[m]['recall_macro'] for m in models]
        }
        df = pd.DataFrame(metrics_data, index=models)
        df.plot(kind='bar', ax=ax4, width=0.8, alpha=0.8)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend(loc='lower right')
        ax4.set_ylim(0, 1.15)
        
        plt.suptitle('IoT Device Classification - Training Summary', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'training_summary.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig


def generate_report(results: Dict[str, Dict], comparison_df: pd.DataFrame,
                    output_path: str = 'results/evaluation_report.txt'):
    """
    Generate a text report of the evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        comparison_df: Model comparison DataFrame
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("IoT DEVICE CLASSIFICATION - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        best_model = comparison_df.iloc[0]['Model']
        f.write(f"Best Performing Model: {best_model}\n")
        f.write(f"  Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}\n")
        f.write(f"  F1-Score (Macro): {comparison_df.iloc[0]['F1 (Macro)']:.4f}\n\n")
        
        for model_name, metrics in results.items():
            f.write("="*70 + "\n")
            f.write(f"DETAILED RESULTS: {model_name}\n")
            f.write("="*70 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(metrics['classification_report'])
            f.write("\n\n")
    
    print(f"Report saved to: {output_path}")
