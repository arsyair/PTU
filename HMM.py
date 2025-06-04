import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.metrics import classification_report

DATA_PATH = "data"
N_MFCC = 13
SR = 16000
N_COMPONENTS = 4  # Jumlah state HMM

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T  # shape: [frame, fitur]

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
        X = np.vstack(features_list)              # semua mfcc digabung
        lengths = [len(x) for x in features_list] # panjang tiap sample
        model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type='diag', n_iter=1000)
        model.fit(X, lengths)
        models[label] = model
        print(f"Trained HMM for label: {label}")
    return models

def classify_hmm(sample_mfcc, models):
    scores = {}
    for label, model in models.items():
        try:
            scores[label] = model.score(sample_mfcc)
        except:
            scores[label] = -np.inf
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
