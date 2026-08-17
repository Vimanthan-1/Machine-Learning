import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

MODEL_NAME = "microsoft/wavlm-base-plus"

class FoundationFeatureExtractor:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print(f"Loading Foundation Model: {MODEL_NAME}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        self.model = WavLMModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def extract_features(self, audio_path):
        try:
            # WavLM expects 16kHz audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Truncate to a maximum of 10 seconds to avoid memory allocation errors (OOM)
            max_samples = 16000 * 10
            if len(y) > max_samples:
                y = y[:max_samples]
            
            # Process to tensors
            inputs = self.processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_values)
                
            # Take the mean over the sequence length (time axis)
            hidden_states = outputs.last_hidden_state # Shape: (1, seq_len, hidden_size)
            mean_embeds = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            return mean_embeds
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {audio_path}: {e}")
            return None

def extract_features_from_df(df, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        data = np.load(cache_path)
        return data['X'], data['y']
        
    print(f"Extracting Foundation features for {len(df)} files...")
    extractor = FoundationFeatureExtractor.get_instance()
    
    X = []
    y = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['file_path']
        label = row['label']
        feats = extractor.extract_features(file_path)
        if feats is not None:
            X.append(feats)
            y.append(label)
            
    X = np.array(X)
    y = np.array(y)
    
    if cache_path:
        print(f"Saving extracted features to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, X=X, y=y)
        
    return X, y

if __name__ == "__main__":
    from src.data_loader import build_dataset_index
    print("Testing Foundation Feature Extraction...")
    df = build_dataset_index()
    df_subset = df.head(5) # Just test on 5 samples
    X, y = extract_features_from_df(df_subset)
    print(f"Extracted {len(X)} features with shape {X.shape}")
