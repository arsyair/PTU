import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.metrics import classification_report

DATA_PATH = "data"
N_MFCC = 13
SR = 16000
N_COMPONENTS = 6  # Coba tingkatkan komponen HMM
N_ITER = 1000

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    # 1. Pre-emphasis
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # 2. Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 3. Normalisasi amplitudo
    if np.max(np.abs(y)) != 0:
        y = y / np.max(np.abs(y))

    # 4. Ekstraksi MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # 5. Ekstraksi delta dan delta-delta MFCC
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Gabungkan fitur mfcc, delta, dan delta2
    combined = np.vstack([mfcc, delta, delta2])

    return combined.T  # shape: [frames, fitur]

def load_dataset(split):
    dataset = {}
    path = os.path.join(DATA_PATH, split)
    for label in os.listdir(path):
        label_dir = os.path.join(path, label)
        if not os.path.isdir(label_dir):
            continue
        dataset[label] = []
        for fname in os.listdir(label_dir):
            if fname.endswith(".wav"):
                fpath = os.path.join(label_dir, fname)
                mfcc = extract_mfcc(fpath)
                dataset[label].append(mfcc)
    return dataset

def train_hmms(train_data):
    models = {}
    for label, features_list in train_data.items():
        X = np.vstack(features_list)
        lengths = [len(x) for x in features_list]
        model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type='diag',
                                n_iter=N_ITER, random_state=42)
        model.fit(X, lengths)
        models[label] = model
        print(f"✅ Trained HMM for label: {label}")
    return models

def classify_hmm(sample_mfcc, models):
    scores = {}
    for label, model in models.items():
        try:
            scores[label] = model.score(sample_mfcc)
        except Exception as e:
            scores[label] = -np.inf
    # Return label dengan skor tertinggi
    return max(scores, key=scores.get)

def evaluate_hmm(test_data, models):
    y_true, y_pred = [], []
    for label, samples in test_data.items():
        for sample in samples:
            pred = classify_hmm(sample, models)
            y_true.append(label)
            y_pred.append(pred)
            print(f"Actual: {label}, Predicted: {pred}")
    return y_true, y_pred

if __name__ == "__main__":
    print("🔄 Memuat data training dan testing...")
    train_data = load_dataset("train")
    test_data = load_dataset("test")

    print("🔧 Melatih model HMM...")
    models = train_hmms(train_data)

    print("🚀 Evaluasi model...")
    y_true, y_pred = evaluate_hmm(test_data, models)

    print("\n📊 === Evaluation Report ===")
    print(classification_report(y_true, y_pred, digits=4))
