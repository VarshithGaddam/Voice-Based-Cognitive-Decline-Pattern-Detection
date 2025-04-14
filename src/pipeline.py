import os
import pandas as pd
from .preprocess import preprocess_audio
from .feature_extraction import extract_features
from .modeling import detect_patterns
from .visualization import plot_features

def run_pipeline(raw_dir, processed_dir):
    """Orchestrate the full pipeline."""
    try:
        print("Starting pipeline...")
        feature_list = []
        
        # Process each audio file
        for audio_file in os.listdir(raw_dir):
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(raw_dir, audio_file)
                audio, sr, transcript = preprocess_audio(audio_path, processed_dir)
                if audio is not None:
                    features = extract_features(audio, sr, transcript, audio_file)
                    feature_list.append(features)
                else:
                    print(f"Skipping {audio_file}: Preprocessing failed")
        
        # Create feature dataframe
        if not feature_list:
            raise ValueError("No valid audio files processed")
        
        feature_df = pd.DataFrame(feature_list)
        if "sample_id" not in feature_df.columns:
            raise ValueError("sample_id column missing in feature_df")
        feature_df.set_index("sample_id", inplace=True)
        
        # Run modeling
        result_df, scaler, kmeans, iso_forest = detect_patterns(feature_df)
        
        # Visualize
        try:
            plot_features(feature_df, result_df)
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Save results
        os.makedirs("results", exist_ok=True)
        result_df.to_csv("results/results.csv")
        
        print("Pipeline completed. Results saved in 'results/'.")
        return result_df, scaler, kmeans, iso_forest
    
    except Exception as e:
        print(f"Pipeline error: {e}")
        return None, None, None, None

if __name__ == "__main__":
    raw_dir = "data/raw_samples/"
    processed_dir = "data/processed/"
    result_df, _, _, _ = run_pipeline(raw_dir, processed_dir)
    print(result_df)