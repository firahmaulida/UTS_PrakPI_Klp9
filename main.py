import os
import re
import sys
import shutil
import time
import json
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

# ===============================
#  SASTRAWI STEMMER
# ===============================
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi Sastrawi stemmer sekali saja (untuk efisiensi)
factory = StemmerFactory()
sastrawi_stemmer = factory.create_stemmer()

# ===============================
#  ANSI COLOR CODES
# ===============================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_color(text, color):
    """Print text dengan warna"""
    print(f"{color}{text}{Colors.END}")

def print_header(text):
    """Print header dengan style"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*50}")
    print(f"{text:^50}")
    print(f"{'='*50}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

# ===============================
#  PROGRESS BAR
# ===============================
def progress_bar(current, total, bar_length=40, prefix="Progress"):
    """Simple progress bar"""
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = f"{100 * current / total:.1f}%"
    print(f"\r{Colors.CYAN}{prefix}: |{bar}| {percent} ({current}/{total}){Colors.END}", end='', flush=True)
    if current >= total:
        print()

# ===============================
#  STOPWORDS EN + ID
# ===============================
STOPWORDS_EN = {
    "the","a","an","and","or","in","on","of","for","to","is","are","was","were",
    "be","been","has","have","had","this","that","these","those","with","as","by",
    "at","from","but","if","then","else","so","not","it","its","they","he","she",
    "we","you","do","does","did","done","can","could","should","would","will",
    "just","than","such","very"
}
STOPWORDS_ID = {
    "yang","dan","di","ke","dari","pada","untuk","dengan","ini","itu","kami","kita",
    "akan","sudah","belum","juga","ada","tidak","iya","atau","sebagai","karena",
    "oleh","mereka","saya","dia","serta","adalah","dapat","bisa","agar","hingga",
    "kembali","lebih","masih","telah","harus","bukan","dalam","antara","para",
    "setelah","tanpa","sementara","sebelum","sesudah","bagi","sehingga"
}
STOPWORDS = STOPWORDS_EN | STOPWORDS_ID

# ===============================
#  REGEX PATTERN
# ===============================
WORD_RE = re.compile(r"[a-zA-Z]+")

# ===============================
#  SASTRAWI STEMMING (PENGGANTI SIMPLE_STEM)
# ===============================
def simple_stem(word: str) -> str:
    """Stemming menggunakan Sastrawi untuk Bahasa Indonesia"""
    try:
        # Sastrawi stemmer sudah handle lowercase dan cleaning
        stemmed = sastrawi_stemmer.stem(word.lower())
        return stemmed
    except Exception as e:
        # Jika ada error, return kata asli
        return word.lower()

# ===============================
# 🧹 NORMALISASI TEKS
# ===============================
def normalize(text: str) -> str:
    """Membersihkan dan normalisasi teks"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ===============================
#  PREPROCESS TOKEN
# ===============================
def preprocess(word: str) -> str:
    """Preprocessing token: lowercase, stopword removal, stemming"""
    w = word.lower()
    if len(w) < 2 or len(w) > 30 or w in STOPWORDS:
        return ""
    w = simple_stem(w)
    if w in STOPWORDS or len(w) < 2:
        return ""
    return w

# ===============================
#  LOAD SEMUA FILE TEKS
# ===============================
def load_all_texts_with_paths(dataset_dir="dataset") -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Membaca semua file teks dalam folder dataset
    Return: (list_of_raw_texts, list_of_paths, dataset_stats)
    """
    all_docs = []
    all_paths = []
    dataset_stats = defaultdict(int)
    
    if not os.path.exists(dataset_dir):
        print_error(f"Folder '{dataset_dir}' tidak ditemukan!")
        return [], [], {}
    
    print_info(f"Memindai folder: {dataset_dir}")
    
    for root, _, files in os.walk(dataset_dir):
        for file in sorted(files):
            if file.endswith(('.txt', '.csv')):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if not content:
                            continue
                        all_docs.append(content)
                        all_paths.append(path)
                        
                        # Track dataset statistics
                        dataset_name = os.path.basename(root)
                        dataset_stats[dataset_name] += 1
                        
                except Exception as e:
                    print_warning(f"Gagal membaca {path}: {e}")
    
    return all_docs, all_paths, dict(dataset_stats)

# ===============================
#  BANGUN INVERTED INDEX
# ===============================
def build_inverted_index_hash(docs):
    """Membangun inverted index dengan hash table"""
    inverted = defaultdict(lambda: defaultdict(int))
    total_docs = len(docs)
    
    print_info(f"Memproses {total_docs} dokumen untuk inverted index...")
    
    for doc_id, text in enumerate(docs, start=1):
        text = normalize(text)
        words = WORD_RE.findall(text)
        for word in words:
            token = preprocess(word)
            if token:
                inverted[token][doc_id] += 1
        
        if doc_id % 10 == 0:
            progress_bar(doc_id, total_docs, prefix="Indexing")
    
    progress_bar(total_docs, total_docs, prefix="Indexing")
    return inverted

def write_outputs(inverted, vocab_file="vocabulary_hash.txt", postings_file="postings_hash.txt"):
    """Menulis vocabulary dan postings ke file"""
    terms = sorted(inverted.keys())
    
    with open(vocab_file, "w", encoding="utf-8") as vf:
        vf.write("\n".join(terms))
    
    with open(postings_file, "w", encoding="utf-8") as pf:
        for term in terms:
            postings = ",".join(f"{doc}:{freq}" for doc, freq in sorted(inverted[term].items()))
            pf.write(f"{term} -> {postings}\n")
    
    print_success(f"Vocabulary: {len(terms)} unique terms")

# ===============================
#  WHOOSH INDEX
# ===============================
WHOOSH_DIR = "whoosh_index"

def build_whoosh_index(docs: List[str], paths: List[str], recreate=True):
    """Membangun Whoosh index untuk fast retrieval"""
    if recreate and os.path.exists(WHOOSH_DIR):
        shutil.rmtree(WHOOSH_DIR)
    if not os.path.exists(WHOOSH_DIR):
        os.mkdir(WHOOSH_DIR)

    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        path=ID(stored=True),
        content=TEXT(stored=False)
    )
    
    ix = index.create_in(WHOOSH_DIR, schema)
    writer = ix.writer()
    
    total = len(docs)
    print_info(f"Membangun Whoosh index untuk {total} dokumen...")
    
    for i, (doc_text, path) in enumerate(zip(docs, paths)):
        writer.add_document(doc_id=str(i), path=path, content=doc_text)
        if (i + 1) % 10 == 0:
            progress_bar(i + 1, total, prefix="Whoosh Index")
    
    progress_bar(total, total, prefix="Whoosh Index")
    writer.commit()
    return ix

def whoosh_search(ix, raw_query: str, top_k=200):
    """Pencarian dengan Whoosh (candidate retrieval)"""
    if ix is None:
        return []
    
    with ix.searcher() as searcher:
        parser = QueryParser("content", schema=ix.schema)
        q = re.sub(r"[^a-zA-Z\s]", " ", raw_query)
        q = re.sub(r"\s+", " ", q).strip()
        
        if not q:
            return []
        
        try:
            whoosh_q = parser.parse(q)
            hits = searcher.search(whoosh_q, limit=top_k)
            cand = [int(hit["doc_id"]) for hit in hits]
        except Exception as e:
            print_warning(f"Whoosh search error: {e}")
            return []
    
    return cand

# ===============================
#  TF-IDF BUILD
# ===============================
def build_tfidf_from_texts(raw_texts: List[str]):
    """Membangun TF-IDF matrix menggunakan CountVectorizer dan TfidfTransformer"""
    print_info(f"Membangun TF-IDF matrix dari {len(raw_texts)} dokumen...")
    
    norm_texts = [normalize(t) for t in raw_texts]
    
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(norm_texts)
    
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    
    print_success(f"TF-IDF matrix: {X_tfidf.shape[0]} docs × {X_tfidf.shape[1]} terms")
    
    return vectorizer, tfidf_transformer, X_tfidf, norm_texts

# ===============================
#  RERANK WITH COSINE SIMILARITY
# ===============================
def rerank_query_with_tfidf(query: str, vectorizer, tfidf_transformer, X_tfidf, 
                             candidate_indices: List[int], top_k=5):
    """Reranking dengan cosine similarity menggunakan TF-IDF"""
    q_norm = normalize(query)
    if not q_norm:
        return []

    q_counts = vectorizer.transform([q_norm])
    q_tfidf = tfidf_transformer.transform(q_counts)

    if not candidate_indices:
        # Fallback: compute against all docs
        sims = cosine_similarity(q_tfidf, X_tfidf).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in top_idx]

    # Compute similarity only for candidates
    import numpy as np
    cand_matrix = X_tfidf[candidate_indices]
    sims = cosine_similarity(q_tfidf, cand_matrix).flatten()
    paired = [(candidate_indices[i], float(sims[i])) for i in range(len(candidate_indices))]
    paired.sort(key=lambda x: x[1], reverse=True)
    
    return paired[:top_k]

# ===============================
#  GET DOCUMENT SNIPPET
# ===============================
def get_snippet(text: str, query: str, max_length=200) -> str:
    """Mengambil snippet dokumen yang relevan dengan query"""
    text_lower = text.lower()
    query_lower = query.lower()
    query_words = query_lower.split()
    
    # Cari posisi kata pertama dari query
    best_pos = -1
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1:
            best_pos = pos
            break
    
    if best_pos == -1:
        # Tidak ditemukan, ambil dari awal
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # Ambil context di sekitar posisi
    start = max(0, best_pos - 100)
    end = min(len(text), best_pos + max_length)
    snippet = text[start:end]
    
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet

# ===============================
#  EXPORT HASIL PENCARIAN
# ===============================
def export_results(query: str, results: List[Tuple[int, float]], paths: List[str], 
                   raw_texts: List[str], filename="search_results.txt"):
    """Export hasil pencarian ke file"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"HASIL PENCARIAN\n")
            f.write(f"Query: {query}\n")
            f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            
            for rank, (doc_idx, score) in enumerate(results, start=1):
                file_path = paths[doc_idx] if doc_idx < len(paths) else "Unknown"
                snippet = get_snippet(raw_texts[doc_idx], query, max_length=300)
                
                f.write(f"#{rank} | Score: {score:.4f}\n")
                f.write(f"Path: {file_path}\n")
                f.write(f"Snippet: {snippet}\n")
                f.write(f"{'-'*70}\n\n")
        
        print_success(f"Hasil disimpan ke: {filename}")
    except Exception as e:
        print_error(f"Gagal export hasil: {e}")

# ===============================
#  DISPLAY STATISTICS
# ===============================
def display_statistics(dataset_stats: Dict[str, int], inverted, X_tfidf):
    """Menampilkan statistik dataset"""
    print_header("STATISTIK DATASET")
    
    print(f"{Colors.BOLD}Dataset Breakdown:{Colors.END}")
    total_docs = sum(dataset_stats.values())
    for dataset_name, count in sorted(dataset_stats.items()):
        percentage = (count / total_docs * 100) if total_docs > 0 else 0
        print(f"  • {dataset_name}: {count} dokumen ({percentage:.1f}%)")
    
    print(f"\n{Colors.BOLD}Index Statistics:{Colors.END}")
    print(f"  • Total dokumen: {total_docs}")
    print(f"  • Unique terms: {len(inverted)}")
    print(f"  • TF-IDF matrix size: {X_tfidf.shape[0]} × {X_tfidf.shape[1]}")
    print(f"  • Stemmer: Sastrawi (Indonesian)")
    print()

# ===============================
#  CLI MAIN MENU
# ===============================
def print_main_menu():
    """Print menu utama dengan style"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════╗")
    print("║   INFORMATION RETRIEVAL SYSTEM - Enhanced      ║")
    print("║           with Sastrawi Stemmer                ║")
    print("╚════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    print(f"{Colors.CYAN}[1]{Colors.END} Load & Index Dataset")
    print(f"{Colors.CYAN}[2]{Colors.END} Search Query")
    print(f"{Colors.CYAN}[3]{Colors.END} Show Statistics")
    print(f"{Colors.CYAN}[4]{Colors.END} Export Last Results")
    print(f"{Colors.CYAN}[5]{Colors.END} Exit")
    print(f"{Colors.BOLD}{'─'*50}{Colors.END}")

def main_cli():
    """Main CLI loop"""
    # State variables
    raw_texts = []
    paths = []
    dataset_stats = {}
    vectorizer = None
    tfidf_transformer = None
    X_tfidf = None
    ix = None
    inverted = {}
    last_results = []
    last_query = ""
    
    # Welcome message
    print_color("\n🎉 Selamat datang di Information Retrieval System!", Colors.BOLD)
    print_info("Sistem pencarian dokumen dengan Whoosh + TF-IDF + Cosine Similarity")
    print_success("✨ Menggunakan Sastrawi Stemmer untuk Bahasa Indonesia")
    
    while True:
        try:
            print_main_menu()
            choice = input(f"{Colors.YELLOW}➤ Pilih menu: {Colors.END}").strip()
            
            if choice == "1":
                print_header("LOAD & INDEX DATASET")
                dataset_dir = input("📁 Path folder dataset (tekan Enter untuk 'dataset'): ").strip() or "dataset"
                
                start_time = time.time()
                
                # Load texts
                raw_texts, paths, dataset_stats = load_all_texts_with_paths(dataset_dir)
                
                if len(raw_texts) == 0:
                    print_error("Tidak ada dokumen ditemukan!")
                    continue
                
                print_success(f"Total dokumen terbaca: {len(raw_texts)}")
                
                # Build inverted index
                print("\n" + "─"*50)
                inverted = build_inverted_index_hash(raw_texts)
                write_outputs(inverted)
                
                # Build Whoosh index
                print("\n" + "─"*50)
                norm_texts = [normalize(t) for t in raw_texts]
                ix = build_whoosh_index(norm_texts, paths, recreate=True)
                
                # Build TF-IDF
                print("\n" + "─"*50)
                vectorizer, tfidf_transformer, X_tfidf, _ = build_tfidf_from_texts(raw_texts)
                
                elapsed = time.time() - start_time
                print(f"\n{Colors.GREEN}{'='*50}")
                print(f"✨ INDEXING SELESAI dalam {elapsed:.2f} detik")
                print(f"{'='*50}{Colors.END}")
                
            elif choice == "2":
                if ix is None or vectorizer is None or X_tfidf is None:
                    print_error("Index belum dibuat! Pilih menu [1] terlebih dahulu.")
                    continue
                
                print_header("PENCARIAN DOKUMEN")
                query = input(f"{Colors.YELLOW}🔍 Masukkan query: {Colors.END}").strip()
                
                if not query:
                    print_warning("Query kosong!")
                    continue
                
                start_time = time.time()
                
                # Whoosh candidate retrieval
                print_info("Mencari kandidat dengan Whoosh...")
                candidates = whoosh_search(ix, query, top_k=200)
                print_info(f"Kandidat ditemukan: {len(candidates)}")
                
                # Rerank with TF-IDF
                print_info("Ranking dengan cosine similarity...")
                top5 = rerank_query_with_tfidf(query, vectorizer, tfidf_transformer, 
                                               X_tfidf, candidates, top_k=5)
                
                elapsed = time.time() - start_time
                
                # Display results
                if not top5:
                    print_warning("Tidak ada hasil yang relevan.")
                    continue
                
                print(f"\n{Colors.GREEN}{'='*50}")
                print(f"🎯 TOP 5 HASIL PENCARIAN (dalam {elapsed:.3f}s)")
                print(f"{'='*50}{Colors.END}\n")
                
                for rank, (doc_idx, score) in enumerate(top5, start=1):
                    file_path = paths[doc_idx] if doc_idx < len(paths) else "Unknown"
                    dataset_name = os.path.basename(os.path.dirname(file_path))
                    snippet = get_snippet(raw_texts[doc_idx], query, max_length=150)
                    
                    print(f"{Colors.BOLD}{Colors.CYAN}#{rank} | Score: {score:.4f}{Colors.END}")
                    print(f"{Colors.YELLOW}📁 Dataset: {dataset_name}{Colors.END}")
                    print(f"📄 {os.path.basename(file_path)}")
                    print(f"💬 {snippet}")
                    print(f"{Colors.BLUE}{'─'*50}{Colors.END}\n")
                
                last_results = top5
                last_query = query
                
            elif choice == "3":
                if not dataset_stats:
                    print_warning("Belum ada data. Load dataset terlebih dahulu.")
                    continue
                display_statistics(dataset_stats, inverted, X_tfidf)
                
            elif choice == "4":
                if not last_results:
                    print_warning("Belum ada hasil pencarian untuk di-export.")
                    continue
                
                filename = input("💾 Nama file output (tekan Enter untuk 'search_results.txt'): ").strip()
                filename = filename or "search_results.txt"
                
                export_results(last_query, last_results, paths, raw_texts, filename)
                
            elif choice == "5":
                print_color("\n👋 Terima kasih telah menggunakan IR System!", Colors.BOLD)
                print_info("Sampai jumpa!")
                break
                
            else:
                print_error("Pilihan tidak valid! Pilih 1-5.")
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}⚠️  Program dihentikan oleh user.{Colors.END}")
            break
        except Exception as e:
            print_error(f"Terjadi error: {e}")
            print_info("Silakan coba lagi atau laporkan bug ini.")

if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Keluar dari program.{Colors.END}")
        sys.exit(0)