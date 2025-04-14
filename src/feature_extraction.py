import librosa
import nltk
import spacy
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt", quiet=True)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spacy model: {e}")
    nlp = None
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_features(audio, sr, transcript, sample_id):
    """Extract audio and NLP features for cognitive decline detection."""
    features = {"sample_id": sample_id}
    
    try:
        # Audio features
        pauses = librosa.effects.split(audio, top_db=20)
        pause_durations = [(end - start) / sr for start, end in pauses]
        features["pause_count"] = len(pause_durations)
        features["pause_avg_duration"] = np.mean(pause_durations) if pause_durations else 0
        
        # Speech rate
        words = word_tokenize(transcript)
        duration = len(audio) / sr
        features["speech_rate"] = len(words) / (duration / 60) if duration > 0 else 0
        
        # Pitch variability (skipped)
        features["pitch_variability"] = 0.0
        
        # NLP features
        hesitations = re.findall(r"\b(uh|um|er|ah)\b", transcript, re.IGNORECASE)
        features["hesitation_count"] = len(hesitations)
        
        tokens = [w.lower() for w in words if w.isalpha()]
        features["lexical_diversity"] = len(set(tokens)) / len(tokens) if tokens else 0
        
        # Sentence completion
        if nlp:
            doc = nlp(transcript)
            incomplete_sentences = sum(1 for sent in doc.sents if not any(t.dep_ == "ROOT" for t in sent))
            features["incomplete_sentences"] = incomplete_sentences
        else:
            features["incomplete_sentences"] = 0
        
        # Semantic similarity
        sentences = sent_tokenize(transcript)
        if len(sentences) > 1:
            embeddings = embedder.encode(sentences)
            similarities = np.dot(embeddings, embeddings.T) / (
                np.linalg.norm(embeddings, axis=1)[:, None] * np.linalg.norm(embeddings, axis=1)
            )
            features["semantic_similarity"] = np.mean(similarities[np.triu_indices(len(similarities), k=1)])
        else:
            features["semantic_similarity"] = 1.0
    
    except Exception as e:
        print(f"Error extracting features for {sample_id}: {e}")
        features.update({
            "pause_count": 0,
            "pause_avg_duration": 0,
            "speech_rate": 0,
            "hesitation_count": 0,
            "lexical_diversity": 0,
            "incomplete_sentences": 0,
            "semantic_similarity": 1.0
        })
    
    return features

if __name__ == "__main__":
    try:
        audio, sr = librosa.load("../data/processed/sample1.wav", sr=16000)
        with open("../data/processed/sample1.txt", "r") as f:
            transcript = f.read()
        features = extract_features(audio, sr, transcript, "sample1.wav")
        print(features)
    except FileNotFoundError:
        print("Test files not found.")