# Information Retrieval System (Enhanced with Sastrawi Stemmer)

Sistem ini adalah **prototipe Information Retrieval (IR)** berbasis **Python**, yang memungkinkan pengguna untuk:

- Melakukan **pencarian teks** dari kumpulan dokumen,
- Menggunakan **Sastrawi Stemmer** untuk pemrosesan Bahasa Indonesia,
- Menggunakan **Whoosh** untuk _fast candidate retrieval_,
- Melakukan **ranking hasil pencarian** dengan **TF-IDF** dan **cosine similarity**.

---

## Fitur Utama

| Fitur                          | Deskripsi                                                                                 |
| :----------------------------- | :---------------------------------------------------------------------------------------- |
| **Preprocessing Teks**         | Normalisasi, tokenisasi, penghapusan stopword (EN+ID), dan stemming menggunakan Sastrawi. |
| **Inverted Index**             | Membangun _vocabulary_ dan _posting list_ untuk seluruh dokumen.                          |
| **Whoosh Search Engine**       | Menyediakan pencarian cepat terhadap dokumen menggunakan query sederhana.                 |
| **TF-IDF + Cosine Similarity** | Menghitung kesamaan antara query dan dokumen untuk _re-ranking_ hasil pencarian.          |
| **Statistik Dataset**          | Menampilkan jumlah dokumen, istilah unik, dan ukuran matriks TF-IDF.                      |
| **Export Hasil Pencarian**     | Menyimpan hasil pencarian beserta _snippet_ teks ke file `.txt`.                          |
| **CLI Interaktif & Berwarna**  | Antarmuka command-line dengan tampilan warna dan progress bar.                            |

---

## Arsitektur Sistem

```
Dataset Folder ─► Preprocessing ─► Inverted Index
                               │
                               ├──► Whoosh Index  ─► Candidate Retrieval
                               │
                               └──► TF-IDF Model  ─► Cosine Similarity Ranking
```

---

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/username/information-retrieval-sastrawi.git
cd information-retrieval-sastrawi
```

### 2. Buat Virtual Environment (opsional)

```bash
python -m venv venv
source venv/bin/activate       # untuk Linux/Mac
venv\Scripts\activate          # untuk Windows
```

### 3. Instal Dependencies

Pastikan Python versi ≥3.9 sudah terpasang.
Kemudian jalankan:

```bash
pip install -r requirements.txt
```

Atau install secara manual:

```bash
pip install sastrawi whoosh scikit-learn pandas
```

---

## Struktur Folder

```
project_root/
│
├── dataset/                # Folder utama berisi dokumen .txt atau .csv
│   ├── kategori1/
│   │   ├── doc1.txt
│   │   ├── doc2.txt
│   │   └── ...
│   └── kategori2/
│       └── ...
│
├── whoosh_index/           # Folder hasil indexing Whoosh
│
├── vocabulary_hash.txt     # Output vocabulary terms
├── postings_hash.txt       # Output posting list
├── search_results.txt      # Hasil ekspor pencarian
│
├── main.py                 # Script utama (program CLI)
└── requirements.txt        # Daftar library yang dibutuhkan
```

---

## Cara Menjalankan Program

Jalankan program utama:

```bash
python main.py
```

Kemudian pilih menu di CLI interaktif:

| Menu  | Deskripsi            |
| ----- | -------------------- |
| `[1]` | Load & Index Dataset |
| `[2]` | Search Query         |
| `[3]` | Show Statistics      |
| `[4]` | Export Last Results  |
| `[5]` | Exit Program         |

---

## Contoh Penggunaan

### 1. Load & Index Dataset

Masukkan path folder dataset (default: `dataset/`).
Sistem akan:

- Membaca seluruh file `.txt` dan `.csv`,
- Membuat **inverted index**,
- Membangun **Whoosh index**, dan
- Menghasilkan **TF-IDF matrix**.

### 2. Search Query

Masukkan query, contoh:

```
Masukkan query: pemanfaatan air bersih di pedesaan
```

Sistem akan menampilkan **5 dokumen paling relevan** berdasarkan cosine similarity.

### 3. Show Statistics

Menampilkan rincian dataset dan jumlah istilah unik, misalnya:

```
Dataset Breakdown:
  • pedesaan: 15 dokumen (60%)
  • perkotaan: 10 dokumen (40%)

Index Statistics:
  • Total dokumen: 25
  • Unique terms: 1345
  • TF-IDF matrix size: 25 × 1345
  • Stemmer: Sastrawi (Indonesian)
```

### 4. Export Hasil

Simpan hasil pencarian ke file:

```
Nama file output (tekan Enter untuk 'search_results.txt'):
```

Output file berisi path dokumen, skor, dan snippet konten yang relevan.

---

## Teknologi yang Digunakan

| Library          | Fungsi                                     |
| ---------------- | ------------------------------------------ |
| **Sastrawi**     | Stemming Bahasa Indonesia                  |
| **Whoosh**       | Full-text indexing & search engine         |
| **scikit-learn** | TF-IDF vectorization dan cosine similarity |
| **pandas**       | Membaca file CSV dataset                   |
| **re (Regex)**   | Pembersihan dan tokenisasi teks            |
| **ANSI Colors**  | Memberikan warna pada output CLI           |

---

## Contoh Cuplikan Kode

```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

sentence = "Pemerintah sedang mempersiapkan kebijakan baru untuk pengelolaan air."
print(stemmer.stem(sentence))
# Output: "perintah sedang siap kebijakan baru untuk kelola air"
```

---

## Kelebihan Proyek Ini

Mendukung **Bahasa Indonesia penuh**
Menggunakan **kombinasi pencarian cepat (Whoosh)** dan **ranking berbasis relevansi (TF-IDF)**
Desain **modular dan terstruktur**
**Progress bar dan warna CLI** membuat pengalaman pengguna lebih informatif
Cocok untuk **riset Information Retrieval atau Tugas Akhir**

---

## Pengembang

**Nama Proyek**: Information Retrieval System with Sastrawi Stemmer
**Mata Kuliah**: Kecerdasan Buatan / Information Retrieval Project

**Pengembang:**

Firah Maulida — NPM: 2308107010034

Zalvia Inasya Zulna — NPM: 2308107010041

Nadia Maghdalena — NPM: 2308107010045

---

## Lisensi

Proyek ini bersifat open-source untuk keperluan akademik dan pembelajaran.

---
