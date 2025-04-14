from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def detect_patterns(feature_df, n_clusters=2):
    """Apply unsupervised learning to detect cognitive decline patterns."""
    try:
        # Define feature columns
        feature_cols = [
            "pause_count", "pause_avg_duration", "speech_rate",
            "hesitation_count", "lexical_diversity", "incomplete_sentences",
            "semantic_similarity"
        ]
        missing_cols = [col for col in feature_cols if col not in feature_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(feature_df[feature_cols])
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Anomaly detection
        iso_forest = IsolationForest(contamination=0.3, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)
        
        # Combine results
        result_df = feature_df.copy()
        result_df["cluster"] = clusters
        result_df["anomaly"] = anomalies  # -1 for anomaly, 1 for normal
        result_df["anomaly_score"] = anomaly_scores
        result_df["risk_score"] = np.where(anomalies == -1, 0.8, 0.2)
        
        return result_df, scaler, kmeans, iso_forest
    except Exception as e:
        print(f"Error in detect_patterns: {e}")
        result_df = feature_df.copy()
        result_df["cluster"] = 0
        result_df["anomaly"] = 1
        result_df["anomaly_score"] = 0.0
        result_df["risk_score"] = 0.2
        return result_df, None, None, None