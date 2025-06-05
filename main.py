import os
import numpy as np
import librosa
import sounddevice as sd
from HMM import train_hmms, classify_hmm
from DTW import classify_dtw

# Parameter
SR = 16000
DURATION = 2  # detik
N_MFCC = 13
PRE_EMPHASIS = 0.97
DATA_PATH = "data"

# Ekstraksi MFCC, mirip dengan di DTW.py dan HMM.py tapi untuk data rekaman in-memory
def extract_mfcc_from_audio(y, sr=SR):
    y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])
    y, _ = librosa.effects.trim(y, top_db=20)
    if np.max(np.abs(y)) != 0:
        y = y / np.max(np.abs(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T

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
                # pakai extract_mfcc dari DTW.py atau HMM.py juga boleh, tapi pakai fungsi sederhana di main.py aja
                y, sr = librosa.load(fpath, sr=SR)
                mfcc = extract_mfcc_from_audio(y, sr)
                dataset[label].append(mfcc)
    return dataset

def record_audio(duration=DURATION, sr=SR):
    print(f"🎙️ Mulai merekam selama {duration} detik...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.flatten()  # jadi 1D array
    print("🎙️ Rekaman selesai.")
    return audio

if __name__ == "__main__":
    print("🔄 Memuat data training...")
    train_data = load_dataset("train")

    print("🔧 Melatih model HMM...")
    hmm_models = train_hmms(train_data)

    audio = record_audio()

    mfcc = extract_mfcc_from_audio(audio, SR)

    dtw_result = classify_dtw(mfcc, train_data)      # panggil dari dtw.py
    hmm_result = classify_hmm(mfcc, hmm_models)      # panggil dari hmm.py

    print(f"\n🔊 Hasil Prediksi:")
    print(f"  🔹 DTW: {dtw_result}")
    print(f"  🔸 HMM: {hmm_result}")
