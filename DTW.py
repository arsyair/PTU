import os
import librosa
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import classification_report

# ========== PARAMETER ==========
DATA_PATH = "data"  # folder berisi /train dan /test
N_MFCC = 13         # jumlah koefisien MFCC
SR = 16000          # sample rate
# ================================

# Ekstrak MFCC dari satu file
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T  # shape: [frame, fitur]

# Muat semua data dari folder
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

# Hitung jarak DTW antara 2 MFCC
def dtw(x, y):
    dist = cdist(x, y, metric='euclidean')
    N, M = dist.shape
    dp = np.full((N+1, M+1), np.inf)
    dp[0, 0] = 0

    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = dist[i-1, j-1]
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    
    return dp[N, M]

# Klasifikasi 1NN DTW
def classify_dtw(test_sample, train_data):
    min_dist = float('inf')
    predicted_label = None

    for label in train_data:
        for train_sample in train_data[label]:
            distance = dtw(test_sample, train_sample)
            if distance < min_dist:
                min_dist = distance
                predicted_label = label
    return predicted_label

# Evaluasi model
def evaluate_dtw(train_data, test_data):
    y_true = []
    y_pred = []

    for label in test_data:
        for sample in test_data[label]:
            pred = classify_dtw(sample, train_data)
            y_true.append(label)
            y_pred.append(pred)
            print(f"Actual: {label}, Predicted: {pred}")
    
    return y_true, y_pred

# Main Program
if __name__ == "__main__":
    print("🔄 Memuat data training dan testing...")
    train_data = load_dataset("train")
    test_data = load_dataset("test")

    print("✅ Data berhasil dimuat.")
    print("🚀 Mulai klasifikasi DTW...")
    y_true, y_pred = evaluate_dtw(train_data, test_data)

    print("\n📊 === Evaluation Report ===")
    print(classification_report(y_true, y_pred, digits=4))
