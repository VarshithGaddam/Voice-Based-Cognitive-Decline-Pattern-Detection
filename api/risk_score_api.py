from fastapi import FastAPI, UploadFile, File
import pandas as pd
from preprocess import preprocess_audio
from feature_extraction import extract_features
from modeling import detect_patterns

app = FastAPI()

@app.post("/risk_score/")
async def get_risk_score(audio_file: UploadFile = File(...)):
    """Return a risk score for cognitive decline."""
    # Save uploaded file temporarily
    temp_path = f"temp_{audio_file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await audio_file.read())
    
    # Preprocess and extract features
    audio, sr, transcript = preprocess_audio(temp_path, "temp/")
    features = extract_features(audio, sr, transcript)
    
    # Convert to dataframe
    feature_df = pd.DataFrame([features])
    
    # Load pre-trained model (assumed saved)
    # For demo, retrain on dummy data
    dummy_df = pd.DataFrame({
        "pause_count": [2, 5, 3, 8, 1],
        "speech_rate": [120, 100, 110, 90, 130],
        "pitch_variability": [50, 30, 40, 20, 60],
        "hesitation_count": [1, 3, 2, 5, 0]
    })
    _, scaler, _, iso_forest = detect_patterns(dummy_df)
    
    # Score new sample
    X = scaler.transform(feature_df)
    score = iso_forest.decision_function(X)[0]
    risk_score = (1 - (score + 1) / 2)  # Normalize to 0–1 (higher = riskier)
    
    return {"sample_id": audio_file.filename, "risk_score": float(risk_score)}