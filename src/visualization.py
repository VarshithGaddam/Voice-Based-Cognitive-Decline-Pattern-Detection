import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_features(feature_df, result_df):
    """Generate visualizations for features and results."""
    try:
        # Ensure output directory exists
        os.makedirs("results/visualizations", exist_ok=True)
        
        # 1. Boxplot of pause_count by cluster
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="cluster", y="pause_count", data=result_df.reset_index())
        plt.title("Pause Count by Cluster")
        plt.savefig("results/visualizations/pause_count_boxplot.png")
        plt.close()
        
        # 2. Scatter plot of speech_rate vs. pause_count
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="speech_rate", y="pause_count", hue="cluster", size="risk_score", data=result_df.reset_index())
        plt.title("Speech Rate vs. Pause Count")
        plt.savefig("results/visualizations/speech_vs_pause_scatter.png")
        plt.close()
        
        # 3. Boxplot of hesitation_count by cluster
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="cluster", y="hesitation_count", data=result_df.reset_index())
        plt.title("Hesitation Count by Cluster")
        plt.savefig("results/visualizations/hesitation_count_boxplot.png")
        plt.close()
        
        # 4. Scatter plot of lexical_diversity vs. incomplete_sentences
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="lexical_diversity", y="incomplete_sentences", hue="cluster", size="risk_score", data=result_df.reset_index())
        plt.title("Lexical Diversity vs. Incomplete Sentences")
        plt.savefig("results/visualizations/lexical_diversity_scatter.png")
        plt.close()
        
        print("Visualizations saved in results/visualizations/")
    except Exception as e:
        print(f"Error in plot_features: {e}")