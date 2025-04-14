import os
from src.pipeline import run_pipeline

def main():
    """Run the cognitive decline detection pipeline."""
    # Define paths
    raw_dir = "data/raw_samples/"
    processed_dir = "data/processed/"
    
    # Ensure directories exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Run pipeline
    print("Starting pipeline...")
    result_df, scaler, kmeans, iso_forest = run_pipeline(raw_dir, processed_dir)
    print("Pipeline completed. Results saved in 'results/'.")
    print(result_df)
    
    # Optional: Save models (for API use)
    import joblib
    joblib.dump(scaler, "results/scaler.pkl")
    joblib.dump(kmeans, "results/kmeans.pkl")
    joblib.dump(iso_forest, "results/iso_forest.pkl")

if __name__ == "__main__":
    main()