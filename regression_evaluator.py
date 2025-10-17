import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class RegressionEvaluator:
    """
    A comprehensive class for evaluating and visualizing regression model performance.
    """
    
    def __init__(self):
        self.models_metrics = {}
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95']
    
    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """
        Evaluate regression model performance with multiple metrics.
        
        Parameters:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model for display purposes
        
        Returns:
        dict: Dictionary containing all metrics
        """
        try:
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            metrics = {
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
            
            # Store metrics for later comparison
            self.models_metrics[model_name] = metrics
            
            # Print results
            print(f"\n{model_name} Performance Metrics:")
            print(f"{'='*40}")
            print(f'R² Score: {r2:.3f}')
            print(f'MSE: {mse:.3f}')
            print(f'RMSE: {rmse:.3f}')
            print(f'MAE: {mae:.3f}')
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
    
    def plot_actual_vs_predicted(self, y_test, y_pred, model_name):
        """
        Plot actual vs predicted values with regression line.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_name} - Actual vs. Predicted")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals_vs_predicted(self, y_test, y_pred, model_name):
        """
        Plot residuals vs predicted values.
        """
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.3)
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"{model_name} - Residuals vs. Predicted")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals_histogram(self, y_test, y_pred, model_name):
        """
        Plot histogram of residuals.
        """
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title(f"{model_name} - Residuals Distribution")
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_single_model_metrics(self, model_name):
        """
        Plot all metrics for a single model as a bar chart.
        """
        if model_name not in self.models_metrics:
            print(f"Model '{model_name}' not found in stored metrics.")
            return
        
        metrics = self.models_metrics[model_name]
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=self.colors[:len(metric_names)])
        plt.title(f'{model_name} Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score/Error Value')
        plt.xlabel('Metrics')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, figsize=(15, 10)):
        """
        Plot comparison of all stored models across all metrics.
        """
        if not self.models_metrics:
            print("No models stored for comparison.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_list = ['r2', 'mse', 'rmse', 'mae']
        metric_titles = ['R² Score', 'MSE', 'RMSE', 'MAE']
        
        for i, (metric, title) in enumerate(zip(metrics_list, metric_titles)):
            row, col = i // 2, i % 2
            values = [self.models_metrics[model][metric] for model in self.models_metrics.keys()]
            
            bars = axes[row, col].bar(self.models_metrics.keys(), values, 
                                      color=self.colors[:len(self.models_metrics)])
            axes[row, col].set_title(title)
            axes[row, col].set_ylabel('Value')
            
            # Rotate x-axis labels if needed
            if len(self.models_metrics) > 3:
                axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_heatmap_comparison(self):
        """
        Create a heatmap comparing all models across all metrics.
        """
        if not self.models_metrics:
            print("No models stored for comparison.")
            return
        
        # Convert to DataFrame for easier plotting
        df_metrics = pd.DataFrame(self.models_metrics).T
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_metrics, annot=True, cmap='RdYlGn_r', center=0.5, 
                    fmt='.3f', cbar_kws={'label': 'Metric Value'})
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Models')
        plt.tight_layout()
        plt.show()
    
    def plot_radar_chart(self, model_name):
        """
        Create a radar/spider chart for a single model.
        """
        if model_name not in self.models_metrics:
            print(f"Model '{model_name}' not found in stored metrics.")
            return
        
        metrics = self.models_metrics[model_name]
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Number of variables
        N = len(categories)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add the first value at the end to close the plot
        values += values[:1]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot the data
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add title
        plt.title(f'{model_name} Performance Metrics', size=16, y=1.1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_radar_chart_comparison(self, figsize=(12, 10)):
        """
        Create a radar/spider chart comparing all models at once.
        """
        if not self.models_metrics:
            print("No models stored for comparison.")
            return
        
        # Get categories (metrics) from the first model
        first_model = list(self.models_metrics.keys())[0]
        categories = list(self.models_metrics[first_model].keys())
        N = len(categories)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot each model
        for i, (model_name, metrics) in enumerate(self.models_metrics.items()):
            values = [metrics[cat] for cat in categories]
            values += values[:1]  # Complete the circle
            
            # Plot the data
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                   color=self.colors[i % len(self.colors)], alpha=0.7)
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Add title
        plt.title('Model Performance Comparison - Radar Chart', size=16, y=1.1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_radar_chart_normalized(self, figsize=(12, 10)):
        """
        Create a normalized radar chart comparing all models.
        Normalizes metrics so they're all on the same scale (0-1).
        """
        if not self.models_metrics:
            print("No models stored for comparison.")
            return
        
        # Get categories (metrics) from the first model
        first_model = list(self.models_metrics.keys())[0]
        categories = list(self.models_metrics[first_model].keys())
        N = len(categories)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Normalize metrics for each category
        normalized_metrics = {}
        for category in categories:
            values = [self.models_metrics[model][category] for model in self.models_metrics.keys()]
            
            # For R², higher is better; for errors, lower is better
            if category == 'r2':
                # R² is already 0-1, but we can normalize to make it more visible
                min_val, max_val = min(values), max(values)
                if max_val - min_val > 0:
                    normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
                else:
                    normalized_values = [0.5 for v in values]  # If all values are the same
            else:
                # For error metrics, invert and normalize (lower error = higher score)
                min_val, max_val = min(values), max(values)
                if max_val - min_val > 0:
                    normalized_values = [(max_val - v) / (max_val - min_val) for v in values]
                else:
                    normalized_values = [0.5 for v in values]  # If all values are the same
            
            for i, model_name in enumerate(self.models_metrics.keys()):
                if model_name not in normalized_metrics:
                    normalized_metrics[model_name] = {}
                normalized_metrics[model_name][category] = normalized_values[i]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot each model
        for i, (model_name, metrics) in enumerate(normalized_metrics.items()):
            values = [metrics[cat] for cat in categories]
            values += values[:1]  # Complete the circle
            
            # Plot the data
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                   color=self.colors[i % len(self.colors)], alpha=0.7)
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set y-axis limits for normalized values
        ax.set_ylim(0, 1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Add title
        plt.title('Normalized Model Performance Comparison', size=16, y=1.1)
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_evaluation(self, y_test, y_pred, model_name):
        """
        Perform comprehensive evaluation with all plots and metrics.
        """
        # Calculate and store metrics
        metrics = self.evaluate_model(y_test, y_pred, model_name)
        
        # Create all plots
        self.plot_actual_vs_predicted(y_test, y_pred, model_name)
        self.plot_residuals_vs_predicted(y_test, y_pred, model_name)
        self.plot_residuals_histogram(y_test, y_pred, model_name)
        self.plot_single_model_metrics(model_name)
        
        return metrics
    
    def get_best_model(self, metric='r2', higher_is_better=True):
        """
        Get the best performing model based on a specific metric.
        
        Parameters:
        metric: Metric to compare ('r2', 'mse', 'rmse', 'mae')
        higher_is_better: True if higher values are better (e.g., R²), False for errors
        """
        if not self.models_metrics:
            print("No models stored for comparison.")
            return None
        
        best_model = None
        best_value = None
        
        for model_name, metrics in self.models_metrics.items():
            value = metrics[metric]
            
            if best_value is None:
                best_value = value
                best_model = model_name
            elif higher_is_better and value > best_value:
                best_value = value
                best_model = model_name
            elif not higher_is_better and value < best_value:
                best_value = value
                best_model = model_name
        
        print(f"Best model by {metric}: {best_model} ({best_value:.3f})")
        return best_model
    
    def print_summary(self):
        """
        Print a summary table of all models and their metrics.
        """
        if not self.models_metrics:
            print("No models stored for comparison.")
            return
        
        # Create summary DataFrame
        df_summary = pd.DataFrame(self.models_metrics).T
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(df_summary.round(3))
        print("="*60)
        
        # Find best models for each metric
        print("\nBEST MODELS BY METRIC:")
        print("-"*30)
        self.get_best_model('r2', higher_is_better=True)
        self.get_best_model('mse', higher_is_better=False)
        self.get_best_model('rmse', higher_is_better=False)
        self.get_best_model('mae', higher_is_better=False) 