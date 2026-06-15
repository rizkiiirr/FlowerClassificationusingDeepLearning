# Klasifikasi Spesies Bunga Menggunakan Pendekatan Machine Learning dan Hybrid Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Deskripsi Proyek
Repositori ini memuat implementasi *pipeline* komparatif untuk tugas pengklasifikasian citra multi-kelas pada spesies bunga lokal. Proyek ini mendemonstrasikan perbandingan kinerja secara empiris antara algoritma *Machine Learning* klasik yang mengandalkan deskriptor bentuk linier (*Histogram of Oriented Gradients*) dengan arsitektur *Deep Learning* yang mengekstraksi representasi semantik tingkat tinggi menggunakan pendekatan *Transfer Learning* (VGG16 dan MobileNetV2) [1]. Proyek ini juga mencakup eksplorasi metode *Hybrid* (CNN-SVM dan CNN-Random Forest) guna mengatasi limitasi pada dataset berdimensi terbatas [2].

Proyek ini dikembangkan sebagai pemenuhan luaran Ujian Akhir Semester (UAS) Mata Kuliah *Machine Learning*, Program Studi Teknologi Informasi, Fakultas Teknik, Universitas Lambung Mangkurat.

## Arsitektur Pemodelan
Proyek ini menguji dan membandingkan 6 skenario pemodelan komputasi yang direpresentasikan ke dalam skrip dan *notebook* terpisah:

**1. Jalur Machine Learning (Baseline)**
* `Model_SVM.py` / `svm_klasifikasi_bunga.py`: Ekstraksi fitur HOG + Klasifikasi *Support Vector Machine* (RBF Kernel).
* `Model_RandomForest.ipynb`: Ekstraksi fitur HOG + Klasifikasi *Random Forest Ensemble*.

**2. Jalur Deep Learning & Transfer Learning**
* `Model_VGG16.ipynb`: *Fine-tuning* arsitektur VGG16 (Lapisan konvolusi dibekukan, penyesuaian pada *Dense Layer* akhir).
* `Model_MobileNetV2.ipynb`: Implementasi arsitektur ringan MobileNetV2 yang dioptimasi untuk komputasi perangkat berdaya rendah.

**3. Jalur Hybrid (Pendekatan Eksperimental Utama)**
* `Model_VGG16_SVM.ipynb`: Pemanfaatan VGG16 sebagai ekstraktor fitur hierarkis murni, dengan lapisan klasifikasi *Softmax* digantikan oleh fungsi optimasi margin *Support Vector Machine* [2].
* `Model_RF_MobileNetV2.ipynb`: Pemanfaatan MobileNetV2 sebagai pengekstraksi fitur yang diumpankan pada algoritma ansambel pohon keputusan (*Random Forest*).

## Spesifikasi Dataset
Dataset merupakan koleksi primer berjumlah **1.440 citra spesimen digital** yang dikategorikan ke dalam 4 (empat) taksonomi flora:
1. `Bintaro`
2. `Melati_Jakarta`
3. `Melati_Jepang`
4. `Tapak_Dara`

**Protokol Preprocessing:**
* **Metode HOG-ML:** Konversi saluran warna RGB ke bentuk *Grayscale*, diiringi kompresi resolusi absolut sebesar 128x128 piksel untuk efisiensi utilitas memori CPU [3].
* **Metode CNN-DL:** Retensi saluran 3 warna RGB dengan penyesuaian dimensi topologi citra sebesar 224x224 piksel untuk memenuhi topologi masukan standar konvolusi ImageNet.

## Struktur Repositori
```text
flowerclassificationusingdeeplearning/
├── Kode Program/
│   ├── Model_MobileNetV2.ipynb
│   ├── Model_RF_MobileNetV2.ipynb
│   ├── Model_RandomForest.ipynb
│   ├── Model_SVM.py
│   ├── Model_VGG16.ipynb
│   ├── Model_VGG16_SVM.ipynb
│   └── svm_klasifikasi_bunga.py
├── 2310817310008_MUHAMMADRIZKIRAMADHAN.pdf (Laporan Analisis Ilmiah Proyek)
├── Presentation Project Flower Classification.pptx (Salindia Presentasi Proyek)
└── README.md
