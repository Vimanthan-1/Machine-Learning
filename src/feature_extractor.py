import numpy as np

def extract_mfcc(audio_path, n_mfcc=13):
    """
    Mock feature extractor until librosa is installed.
    """
    return np.random.rand(n_mfcc)

if __name__ == "__main__":
    print("Testing Feature Extraction Module...")
    real_features = extract_mfcc('dataset/real/sample_01.wav')
    fake_features = extract_mfcc('dataset/fake/sample_01.wav')
    
    print("\n--- Extracted MFCC Features (Mean over time) ---")
    if real_features is not None:
        print(f"Real Audio MFCC Vector Shape: {real_features.shape}")
        print(f"Real Audio Sample Features (First 5): {np.round(real_features[:5], 3)}")
    if fake_features is not None:
        print(f"Fake Audio MFCC Vector Shape: {fake_features.shape}")
        print(f"Fake Audio Sample Features (First 5): {np.round(fake_features[:5], 3)}")
    print("Feature extraction successful!")
