import matplotlib
matplotlib.use('Agg') # Server side plotting setup
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import numpy as np

class DataVisualizer:
    def __init__(self, df, target_col, problem_type='regression'):
        self.df = df
        self.target_col = target_col
        self.problem_type = problem_type
        sns.set_style("whitegrid") # Clean style

    def _get_image_base64(self):
        """Converts plot to Base64 string for HTML"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close() # Important: Close plot to free memory
        graphic = base64.b64encode(image_png)
        return "data:image/png;base64," + graphic.decode('utf-8')

    def plot_distribution(self):
        plt.figure(figsize=(8, 5))
        
        if self.problem_type == 'regression':
            # Histogram for Regression
            sns.histplot(self.df[self.target_col], kde=True, color='#4F46E5', edgecolor='white')
            plt.title(f'Target Distribution: {self.target_col}', fontsize=14)
            plt.xlabel(self.target_col)
            plt.ylabel('Frequency')
        else:
            # Count Plot for Classification (Top 10 classes)
            top_classes = self.df[self.target_col].value_counts().nlargest(10).index
            filtered_data = self.df[self.df[self.target_col].isin(top_classes)]
            
            sns.countplot(x=self.target_col, data=filtered_data, palette='viridis', order=top_classes)
            plt.title(f'Class Distribution: {self.target_col} (Top 10)', fontsize=14)
            plt.xlabel(self.target_col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        return self._get_image_base64()

    def plot_heatmap(self):
        # Select numeric columns only
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # If numeric data is too less, return None
        viz_df = numeric_df.copy()
        if viz_df.shape[1] < 2: return None
        
        # Limit to top 15 correlated features to avoid messy heatmap
        if viz_df.shape[1] > 15 and self.target_col in viz_df.columns:
            corr_with_target = viz_df.corrwith(viz_df[self.target_col]).abs()
            top_features = corr_with_target.sort_values(ascending=False).head(15).index
            viz_df = viz_df[top_features]
        elif viz_df.shape[1] > 15:
             viz_df = viz_df.iloc[:, :15]

        plt.figure(figsize=(10, 8))
        corr = viz_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool)) # Hide upper triangle
        
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    annot_kws={"size": 9})
        plt.title('Correlation Heatmap (Top Features)', fontsize=14)
        
        plt.tight_layout()
        return self._get_image_base64()

    def plot_top_correlations(self):
        # Bar chart showing which features affect the target most
        numeric_df = self.df.select_dtypes(include=[np.number])
        viz_df = numeric_df.copy()
        
        if self.target_col not in viz_df.columns: return None

        corr = viz_df.corrwith(viz_df[self.target_col]).sort_values(ascending=False)
        corr = corr.drop(labels=[self.target_col], errors='ignore')
        
        # Top 10 strongest correlations (positive or negative)
        top_corr = corr.abs().sort_values(ascending=False).head(10)
        top_corr_signed = corr[top_corr.index]
        
        if top_corr_signed.empty: return None

        plt.figure(figsize=(10, 5))
        colors = ['#10B981' if x > 0 else '#EF4444' for x in top_corr_signed.values]
        sns.barplot(x=top_corr_signed.values, y=top_corr_signed.index, palette=colors)
        plt.title('Top 10 Factors Affecting Target', fontsize=14)
        plt.xlabel('Correlation Coefficient (Green=Positive, Red=Negative)')
        
        plt.tight_layout()
        return self._get_image_base64()

    def generate_all(self):
        return {
            'distribution': self.plot_distribution(),
            'heatmap': self.plot_heatmap(),
            'correlations': self.plot_top_correlations()
        }