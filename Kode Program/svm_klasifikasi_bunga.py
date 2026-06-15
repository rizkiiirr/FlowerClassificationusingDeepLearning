# =============================================================================
# KLASIFIKASI GAMBAR BUNGA MENGGUNAKAN SUPPORT VECTOR MACHINE (SVM)
# Mata Kuliah  : Machine Learning
# Dataset      : Bunga Melati Jakarta, Melati Jepang, Bintaro, Tapak Dara
#                (4 kelas × 360 gambar = 1.440 total data)
# Metode       : SVM + HOG Feature Extraction + GridSearchCV
# IMG_SIZE     : 224 × 224 piksel
# =============================================================================

# ─── LIBRARY ──────────────────────────────────────────────────────────────────
import os
import time
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── KONFIGURASI GLOBAL ───────────────────────────────────────────────────────
DATASET_PATH  = "dataset_bunga"   # Sesuaikan dengan lokasi folder dataset
IMG_SIZE      = (224, 224)        # Ukuran resize gambar (224×224)
RANDOM_STATE  = 42

# HOG: dengan 224×224, pixels_per_cell (14,14) dan (16,16) menghasilkan
# vektor fitur yang proporsional tanpa membebani memori/waktu training.
#   224×224 + (14,14) → ±11.664 fitur
#   224×224 + (16,16) →  ±6.084 fitur
# Bandingkan: 128×128 + (8,8) → ±8.100 fitur

# CLAHE tile grid disesuaikan ke (16,16) agar setiap tile tetap representatif
# pada gambar berukuran lebih besar.
CLAHE_TILE    = (16, 16)

# Kernel Gaussian Blur sedikit diperbesar ke (5,5) karena gambar lebih besar
# sehingga efek blur terasa proporsional (3×3 terlalu lemah di 224×224).
BLUR_KERNEL   = (5, 5)

# ─── WARNA VISUALISASI ────────────────────────────────────────────────────────
COLORS = {
    "primary"   : "#2E8B57",   # hijau daun
    "secondary" : "#87CEEB",   # biru langit
    "accent"    : "#FF6347",   # merah tomat
    "light"     : "#F0FFF0",   # honeydew
    "neutral"   : "#4A4A4A",
}


# ==============================================================================
# BAGIAN 1 — LOAD & EKSPLORASI DATASET
# ==============================================================================
def load_dataset(dataset_path: str):
    """
    Memuat seluruh gambar dari folder dataset.

    Struktur folder yang diharapkan:
        dataset_bunga/
            bintaro/        ← 360 gambar
            melati_jakarta/ ← 360 gambar
            melati_jepang/  ← 360 gambar
            tapak_dara/     ← 360 gambar

    Returns
    -------
    images      : np.ndarray  shape (N, 224, 224) — grayscale
    labels      : np.ndarray  shape (N,)          — integer label
    class_names : list[str]
    """
    images, labels = [], []
    class_names = sorted(os.listdir(dataset_path))   # urutan alfabet agar konsisten

    print(f"\n{'='*60}")
    print(f"  EKSPLORASI DATASET")
    print(f"{'='*60}")
    print(f"  Path dataset  : {os.path.abspath(dataset_path)}")
    print(f"  IMG_SIZE      : {IMG_SIZE[0]}×{IMG_SIZE[1]} piksel")
    print(f"  Kelas ({len(class_names)})    : {class_names}\n")

    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        count_before = len(images)

        for file in sorted(os.listdir(class_folder)):
            img_path = os.path.join(class_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. Resize → 224×224 (INTER_AREA terbaik untuk downscale)
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

            # 2. Konversi ke Grayscale
            #    HOG memanfaatkan gradien intensitas; grayscale mengurangi
            #    dimensi tanpa kehilangan informasi tekstur/tepi bunga.
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            images.append(img_gray)
            labels.append(label_idx)

        jumlah_loaded = len(images) - count_before
        print(f"  [{label_idx}] {class_name:<20} → {jumlah_loaded} gambar dimuat")

    print(f"\n  Total data    : {len(images)} gambar")
    print(f"  Shape array   : {np.array(images).shape}")
    return np.array(images), np.array(labels), class_names


def visualisasi_sampel(images: np.ndarray, labels: np.ndarray, class_names: list):
    """Menampilkan 4 sampel gambar (1 per kelas) beserta distribusi kelas."""

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Eksplorasi Dataset Bunga (224×224)", fontsize=14,
                 fontweight="bold", color=COLORS["neutral"])

    for cls_idx, cls_name in enumerate(class_names):
        idx = np.where(labels == cls_idx)[0][0]
        axes[0, cls_idx].imshow(images[idx], cmap="gray")
        axes[0, cls_idx].set_title(cls_name, fontsize=9, fontweight="bold")
        axes[0, cls_idx].axis("off")

    for ax in axes[1, :]:
        ax.remove()

    ax_dist = fig.add_subplot(2, 1, 2)
    counts  = [np.sum(labels == i) for i in range(len(class_names))]
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#9370DB"]
    bars    = ax_dist.bar(class_names, counts, color=palette,
                          edgecolor="white", linewidth=0.8)
    ax_dist.set_title("Distribusi Jumlah Gambar per Kelas", fontsize=11, fontweight="bold")
    ax_dist.set_ylabel("Jumlah Gambar")
    ax_dist.set_ylim(0, max(counts) + 60)
    for bar, cnt in zip(bars, counts):
        ax_dist.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_dist.set_facecolor(COLORS["light"])
    ax_dist.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("1_eksplorasi_dataset.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Grafik disimpan → 1_eksplorasi_dataset.png")


# ==============================================================================
# BAGIAN 2 — PREPROCESSING DATA
# ==============================================================================
def preprocessing_pipeline(images: np.ndarray):
    """
    Preprocessing tambahan sebelum ekstraksi fitur HOG:
        • CLAHE (tileGridSize=16×16) : pemerataan histogram adaptif,
          meningkatkan kontras lokal pada gambar 224×224.
        • Gaussian Blur kernel (5×5) : meredam noise salt-and-pepper;
          kernel diperbesar dari 3×3 karena resolusi gambar lebih besar.
    """
    processed = []
    # tileGridSize (16,16) → setiap tile berukuran 224/16 = 14×14 piksel,
    # memberikan pemerataan kontras yang representatif di seluruh gambar.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=CLAHE_TILE)

    for img in images:
        img_clahe = clahe.apply(img)
        img_blur  = cv2.GaussianBlur(img_clahe, BLUR_KERNEL, 0)
        processed.append(img_blur)

    return np.array(processed)


def visualisasi_preprocessing(images_raw: np.ndarray, images_proc: np.ndarray,
                               class_names: list, labels: np.ndarray):
    """Perbandingan gambar sebelum dan sesudah preprocessing."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle(
        f"Preprocessing: CLAHE (tile {CLAHE_TILE[0]}×{CLAHE_TILE[1]}) + "
        f"Gaussian Blur ({BLUR_KERNEL[0]}×{BLUR_KERNEL[1]})",
        fontsize=12, fontweight="bold"
    )

    for i in range(4):
        idx = np.where(labels == i)[0][0]
        axes[0, i].imshow(images_raw[idx], cmap="gray")
        axes[0, i].set_title(f"Raw\n{class_names[i]}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(images_proc[idx], cmap="gray")
        axes[1, i].set_title(f"CLAHE+Blur\n{class_names[i]}", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Sebelum", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Sesudah", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("2_preprocessing.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Grafik disimpan → 2_preprocessing.png")


# ==============================================================================
# BAGIAN 3 — EKSTRAKSI FITUR HOG
# ==============================================================================
def extract_hog_features(images: np.ndarray, pixels_per_cell: tuple = (14, 14)):
    """
    Mengekstrak fitur HOG (Histogram of Oriented Gradients).

    Pilihan pixels_per_cell untuk IMG_SIZE (224, 224):
        (14, 14) → grid 16×16 sel  → vektor ≈ 11.664 fitur  [lebih detail]
        (16, 16) → grid 14×14 sel  → vektor ≈  6.084 fitur  [lebih ringan]

    Percobaan 1 & 2 menggunakan (14,14), Percobaan 3 menggunakan (16,16)
    agar ada variasi yang dapat dibandingkan dalam laporan.

    HOG menangkap distribusi lokal arah gradien (tepi & tekstur kelopak bunga),
    memungkinkan SVM membedakan kelas tanpa informasi warna.
    """
    features = []
    for img in images:
        hog_feat = hog(
            img,
            orientations=9,             # 9 bin orientasi (0°–180°)
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),     # normalisasi per blok 2×2 sel
            block_norm="L2-Hys",        # robust terhadap perubahan pencahayaan
            visualize=False,
            feature_vector=True
        )
        features.append(hog_feat)

    features_arr = np.array(features)
    print(f"  [HOG] pixels_per_cell={pixels_per_cell} "
          f"→ panjang vektor fitur: {features_arr.shape[1]:,}")
    return features_arr


def visualisasi_hog(images: np.ndarray, labels: np.ndarray,
                    class_names: list, pixels_per_cell: tuple = (14, 14)):
    """Menampilkan gambar asli vs visualisasi HOG untuk 1 sampel per kelas."""
    from skimage.feature import hog as skimage_hog

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle(
        f"Visualisasi HOG — pixels_per_cell={pixels_per_cell}, "
        f"IMG_SIZE={IMG_SIZE[0]}×{IMG_SIZE[1]}",
        fontsize=11, fontweight="bold"
    )

    for i in range(4):
        idx = np.where(labels == i)[0][0]
        _, hog_img = skimage_hog(
            images[idx],
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
            feature_vector=True
        )
        axes[0, i].imshow(images[idx], cmap="gray")
        axes[0, i].set_title(class_names[i], fontsize=8, fontweight="bold")
        axes[0, i].axis("off")

        axes[1, i].imshow(hog_img, cmap="inferno")
        axes[1, i].set_title("HOG Features", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Gambar Asli", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("HOG", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("3_hog_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Grafik disimpan → 3_hog_features.png")


# ==============================================================================
# BAGIAN 4 — SPLIT DATA
# ==============================================================================
def split_data(features: np.ndarray, labels: np.ndarray, test_size: float = 0.2):
    """
    Membagi data menjadi Training : Testing.
    stratify=labels memastikan distribusi kelas seimbang di kedua split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        stratify=labels,
        random_state=RANDOM_STATE
    )
    n_total = len(features)
    print(f"\n  Pembagian data (test_size={test_size}):")
    print(f"    Training : {X_train.shape[0]} sampel "
          f"({X_train.shape[0]/n_total*100:.0f}%) — fitur: {X_train.shape[1]:,}")
    print(f"    Testing  : {X_test.shape[0]} sampel "
          f"({X_test.shape[0]/n_total*100:.0f}%)")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# BAGIAN 5 — PELATIHAN MODEL SVM + GRID SEARCH
# ==============================================================================
def train_svm(X_train: np.ndarray, y_train: np.ndarray):
    """
    Melatih model SVM dengan Pipeline:
        1. StandardScaler — menormalkan fitur HOG (mean=0, std=1)
           Wajib karena panjang vektor HOG di 224×224 lebih bervariasi.
        2. SVC (kernel RBF & Linear) dengan probability=True
        3. GridSearchCV 5-fold CV → hyperparameter terbaik

    Catatan: dengan vektor fitur ~11.664 dimensi, GridSearchCV membutuhkan
    waktu lebih lama daripada 128×128. Estimasi: 20–45 menit tergantung CPU.
    """
    print("\n  Memulai GridSearchCV (5-fold CV) ...")
    print("  Estimasi waktu: 20–45 menit (tergantung spesifikasi CPU)\n")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            decision_function_shape="ovo",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ])

    param_grid = {
        "svm__kernel" : ["rbf", "linear"],
        "svm__C"      : [0.1, 1, 10, 100],
        "svm__gamma"  : ["scale", 0.01, 0.001]
    }

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,      # pakai semua core CPU
        verbose=1
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  [✓] GridSearchCV selesai dalam {elapsed/60:.1f} menit")
    print(f"  Best Parameters : {grid.best_params_}")
    print(f"  Best CV Score   : {grid.best_score_:.4f} ({grid.best_score_*100:.2f}%)")

    return grid.best_estimator_, grid


# ==============================================================================
# BAGIAN 6 — EVALUASI MODEL
# ==============================================================================
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: list, label: str = ""):
    """
    Mengevaluasi model SVM:
        • Accuracy, Precision, Recall, F1-Score (weighted)
        • Classification Report per kelas
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*60}")
    print(f"  HASIL EVALUASI — {label}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    return {
        "accuracy" : acc, "precision": prec,
        "recall"   : rec, "f1"       : f1,
        "y_pred"   : y_pred, "label" : label
    }


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray,
                          class_names: list, label: str = "",
                          filename: str = "cm.png"):
    """Confusion Matrix heatmap: nilai absolut (kiri) + persentase (kanan)."""
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {label}", fontsize=13, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5, linecolor="white")
    axes[0].set_xlabel("Prediksi", fontweight="bold")
    axes[0].set_ylabel("Aktual", fontweight="bold")
    axes[0].set_title("Jumlah Absolut")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, linecolor="white")
    axes[1].set_xlabel("Prediksi", fontweight="bold")
    axes[1].set_ylabel("Aktual", fontweight="bold")
    axes[1].set_title("Persentase per Kelas")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [✓] Confusion matrix disimpan → {filename}")


# ==============================================================================
# BAGIAN 7 — CROSS VALIDATION
# ==============================================================================
def cross_validation(model, features: np.ndarray,
                     labels: np.ndarray, cv: int = 5):
    """
    Cross-validation 5-fold untuk mengukur generalisasi model secara robust.
    """
    print(f"\n  Cross Validation ({cv}-fold) ...")
    scores = cross_val_score(
        model, features, labels,
        cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"  Skor per fold  : {np.round(scores, 4)}")
    print(f"  Rata-rata      : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def plot_cv_scores(cv_scores: np.ndarray, label: str = ""):
    """Visualisasi skor cross-validation per fold."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bar_colors  = [COLORS["primary"] if s >= cv_scores.mean() else COLORS["accent"]
                   for s in cv_scores]
    bars = ax.bar(fold_labels, cv_scores * 100, color=bar_colors,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(cv_scores.mean() * 100, color="navy", linestyle="--",
               linewidth=1.5, label=f"Rata-rata: {cv_scores.mean()*100:.2f}%")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(f"Cross Validation (5-Fold) — {label}", fontweight="bold")
    ax.legend()
    ax.set_facecolor(COLORS["light"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("5_cross_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Grafik disimpan → 5_cross_validation.png")


# ==============================================================================
# BAGIAN 8 — PERBANDINGAN PERCOBAAN
# ==============================================================================
def plot_perbandingan_percobaan(results: list):
    """
    Grafik batang perbandingan 3 percobaan × 4 metrik (Accuracy, Precision,
    Recall, F1) dalam satu figur untuk laporan.
    """
    metrics     = ["accuracy", "precision", "recall", "f1"]
    labels      = [r["label"] for r in results]
    x           = np.arange(len(labels))
    width       = 0.18
    colors_list = [COLORS["primary"], COLORS["secondary"],
                   COLORS["accent"], "#9370DB"]

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, metric in enumerate(metrics):
        values = [r[metric] * 100 for r in results]
        rects  = ax.bar(x + i * width, values, width,
                        label=metric.capitalize(),
                        color=colors_list[i], edgecolor="white", linewidth=0.8)
        for rect in rects:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.3,
                    f"{rect.get_height():.1f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("Percobaan", fontweight="bold")
    ax.set_ylabel("Nilai Metrik (%)", fontweight="bold")
    ax.set_title("Perbandingan Metrik Evaluasi antar Percobaan SVM\n"
                 f"(IMG_SIZE {IMG_SIZE[0]}×{IMG_SIZE[1]})", fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_facecolor(COLORS["light"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("6_perbandingan_percobaan.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Grafik perbandingan disimpan → 6_perbandingan_percobaan.png")


# ==============================================================================
# BAGIAN 9 — DEMO PREDIKSI GAMBAR BARU
# ==============================================================================
def prediksi_gambar_baru(model, image_path: str, class_names: list,
                          pixels_per_cell: tuple = (14, 14)):
    """
    Melakukan prediksi kelas pada satu gambar baru.
    Gunakan fungsi ini untuk demo live saat presentasi UAS.

    Cara pakai:
        prediksi_gambar_baru(model_terbaik, "foto_bunga.jpg", class_names)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [!] Gambar tidak ditemukan: {image_path}")
        return

    # Terapkan pipeline preprocessing yang sama dengan data training
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_gray    = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=CLAHE_TILE)
    img_clahe   = clahe.apply(img_gray)
    img_blur    = cv2.GaussianBlur(img_clahe, BLUR_KERNEL, 0)

    feat = hog(
        img_blur,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    ).reshape(1, -1)

    pred_idx  = model.predict(feat)[0]
    pred_prob = model.predict_proba(feat)[0]
    pred_name = class_names[pred_idx]

    print(f"\n  [DEMO PREDIKSI]")
    print(f"  File           : {image_path}")
    print(f"  Kelas prediksi : {pred_name} "
          f"(confidence: {pred_prob[pred_idx]*100:.2f}%)")
    for i, (name, prob) in enumerate(zip(class_names, pred_prob)):
        bar_str = "█" * int(prob * 30)
        print(f"    {name:<20} {prob*100:5.2f}%  {bar_str}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Gambar (224×224)", fontsize=11)
    axes[0].axis("off")

    colors_prob = [COLORS["primary"] if i == pred_idx else COLORS["secondary"]
                   for i in range(len(class_names))]
    axes[1].barh(class_names, pred_prob * 100, color=colors_prob, edgecolor="white")
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title(f"Prediksi: {pred_name}", fontsize=11,
                      fontweight="bold", color=COLORS["primary"])
    axes[1].set_xlim(0, 115)
    for i, prob in enumerate(pred_prob):
        axes[1].text(prob * 100 + 1, i, f"{prob*100:.1f}%",
                     va="center", fontsize=9)
    axes[1].set_facecolor(COLORS["light"])

    plt.tight_layout()
    plt.savefig("7_demo_prediksi.png", dpi=150, bbox_inches="tight")
    plt.show()
    return pred_name, pred_prob


# ==============================================================================
# BAGIAN 10 — RINGKASAN AKHIR
# ==============================================================================
def print_ringkasan(all_results: list):
    """Tabel ringkasan seluruh percobaan di terminal."""
    print(f"\n{'='*72}")
    print(f"  RINGKASAN SELURUH PERCOBAAN SVM  (IMG_SIZE {IMG_SIZE[0]}×{IMG_SIZE[1]})")
    print(f"{'='*72}")
    header = f"  {'Percobaan':<30} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}"
    print(header)
    print(f"  {'-'*65}")
    for r in all_results:
        print(f"  {r['label']:<30} "
              f"{r['accuracy']*100:>6.2f}% "
              f"{r['precision']*100:>6.2f}% "
              f"{r['recall']*100:>6.2f}% "
              f"{r['f1']*100:>6.2f}%")
    print(f"{'='*72}")

    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"\n  ★ Percobaan terbaik : {best['label']}")
    print(f"    Accuracy          : {best['accuracy']*100:.2f}%")
    print(f"    F1-Score          : {best['f1']*100:.2f}%\n")


# ==============================================================================
# HELPER — satu percobaan penuh
# ==============================================================================
def run_percobaan(pixels_per_cell: tuple, test_size: float, label: str,
                  images_proc: np.ndarray, labels_arr: np.ndarray,
                  class_names: list):
    """Menjalankan pipeline lengkap satu percobaan SVM."""
    print(f"\n\n{'#'*62}")
    print(f"#  {label}")
    print(f"{'#'*62}")

    features = extract_hog_features(images_proc, pixels_per_cell)
    X_train, X_test, y_train, y_test = split_data(features, labels_arr, test_size)
    model, _ = train_svm(X_train, y_train)
    result   = evaluate_model(model, X_test, y_test, class_names, label)

    safe_label = (label.replace(" ", "_")
                       .replace("(", "").replace(")", "")
                       .replace(",", "").replace("×", "x"))
    plot_confusion_matrix(
        y_test, result["y_pred"], class_names, label,
        filename=f"4_confusion_matrix_{safe_label}.png"
    )

    return model, result, features


# ==============================================================================
# MAIN — PIPELINE LENGKAP
# ==============================================================================
if __name__ == "__main__":

    # ── 1. LOAD DATASET ───────────────────────────────────────────────────────
    images_raw, labels_arr, class_names = load_dataset(DATASET_PATH)
    visualisasi_sampel(images_raw, labels_arr, class_names)

    # ── 2. PREPROCESSING ──────────────────────────────────────────────────────
    print(f"\n  Menerapkan preprocessing "
          f"(CLAHE tile {CLAHE_TILE[0]}×{CLAHE_TILE[1]} + "
          f"Gaussian Blur {BLUR_KERNEL[0]}×{BLUR_KERNEL[1]}) ...")
    images_proc = preprocessing_pipeline(images_raw)
    visualisasi_preprocessing(images_raw, images_proc, class_names, labels_arr)

    # ── 3. VISUALISASI HOG ────────────────────────────────────────────────────
    # Gunakan pixels_per_cell (14,14) sebagai referensi visualisasi utama
    visualisasi_hog(images_proc, labels_arr, class_names, pixels_per_cell=(14, 14))

    # ── 4. JALANKAN 3 PERCOBAAN ───────────────────────────────────────────────
    # Variasi percobaan:
    #   • pixels_per_cell (14,14) → vektor ≈ 11.664 fitur (detail)
    #   • pixels_per_cell (16,16) → vektor ≈  6.084 fitur (efisien)
    #   • test_size 70/30 vs 80/20 → pengaruh ukuran training set
    all_results  = []
    all_models   = []
    all_features = []
    all_pcells   = []

    # Percobaan 1: HOG 14×14, split 70/30
    m1, r1, f1 = run_percobaan(
        pixels_per_cell=(14, 14), test_size=0.3,
        label="Percobaan 1 (HOG 14×14, 70-30)",
        images_proc=images_proc, labels_arr=labels_arr, class_names=class_names
    )
    all_results.append(r1); all_models.append(m1)
    all_features.append(f1); all_pcells.append((14, 14))

    # Percobaan 2: HOG 14×14, split 80/20
    m2, r2, f2 = run_percobaan(
        pixels_per_cell=(14, 14), test_size=0.2,
        label="Percobaan 2 (HOG 14×14, 80-20)",
        images_proc=images_proc, labels_arr=labels_arr, class_names=class_names
    )
    all_results.append(r2); all_models.append(m2)
    all_features.append(f2); all_pcells.append((14, 14))

    # Percobaan 3: HOG 16×16, split 80/20
    m3, r3, f3 = run_percobaan(
        pixels_per_cell=(16, 16), test_size=0.2,
        label="Percobaan 3 (HOG 16×16, 80-20)",
        images_proc=images_proc, labels_arr=labels_arr, class_names=class_names
    )
    all_results.append(r3); all_models.append(m3)
    all_features.append(f3); all_pcells.append((16, 16))

    # ── 5. CROSS VALIDATION (model terbaik) ───────────────────────────────────
    best_idx      = int(np.argmax([r["accuracy"] for r in all_results]))
    model_terbaik = all_models[best_idx]
    feat_terbaik  = all_features[best_idx]
    label_terbaik = all_results[best_idx]["label"]

    print(f"\n\n  Menjalankan Cross Validation pada: {label_terbaik}")
    cv_scores = cross_validation(model_terbaik, feat_terbaik, labels_arr, cv=5)
    plot_cv_scores(cv_scores, label_terbaik)

    # ── 6. PERBANDINGAN PERCOBAAN ─────────────────────────────────────────────
    plot_perbandingan_percobaan(all_results)

    # ── 7. RINGKASAN ──────────────────────────────────────────────────────────
    print_ringkasan(all_results)

    # ── 8. DEMO PREDIKSI (aktifkan saat presentasi UAS) ───────────────────────
    # Ganti "contoh_bunga.jpg" dengan path gambar yang ingin diprediksi
    # prediksi_gambar_baru(
    #     model_terbaik, "contoh_bunga.jpg", class_names,
    #     pixels_per_cell=all_pcells[best_idx]
    # )

    print("\n  [✓] Semua selesai. File PNG siap digunakan untuk laporan UAS.")
