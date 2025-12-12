import os

import matplotlib.pyplot as plt
import seaborn as sns


class VisualizerAgent:
    def __init__(self, output_dir="results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_data_overview(self, df, filename="data_overview.png"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Overview', fontsize=16, fontweight='bold')

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        if len(numeric_cols) > 0:
            df[numeric_cols[0]].hist(bins=30, ax=axes[0, 0], edgecolor='black')
            axes[0, 0].set_title(f'Distribution: {numeric_cols[0]}')
            axes[0, 0].set_xlabel(numeric_cols[0])
            axes[0, 0].set_ylabel('Frequency')

            if len(numeric_cols) > 1:
                df.boxplot(column=numeric_cols[:5].tolist(), ax=axes[0, 1])
                axes[0, 1].set_title('Box Plot: Numeric Features')
                axes[0, 1].tick_params(axis='x', rotation=45)

            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                        cmap='coolwarm', ax=axes[1, 0], cbar_kws={'shrink': 0.8})
            axes[1, 0].set_title('Correlation Matrix')

            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                missing_data[missing_data > 0].plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Missing Values per Column')
                axes[1, 1].set_ylabel('Count')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Missing Values',
                                ha='center', va='center', fontsize=14)
                axes[1, 1].set_title('Missing Values')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_metrics(self, metrics, filename="pipeline_metrics.png"):
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        colors = plt.cm.viridis([i / len(metric_names) for i in range(len(metric_names))])
        bars = ax.barh(metric_names, metric_values, color=colors)

        ax.set_xlabel('Value', fontsize=12)
        ax.set_title('Pipeline Metrics', fontsize=16, fontweight='bold')

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_processing_time(self, metrics, filename="processing_time.png"):
        fig, ax = plt.subplots(figsize=(8, 6))

        time_metrics = {k: v for k, v in metrics.items() if 'time' in k}

        if time_metrics:
            labels = list(time_metrics.keys())
            values = list(time_metrics.values())

            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            explode = [0.05] * len(labels)

            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=colors[:len(labels)], explode=explode)
            ax.set_title('Time Distribution', fontsize=16, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath