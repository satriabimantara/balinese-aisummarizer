# balinese-aisummarizer · `satuasummarization`
Artificial Intelligence Models for Balinese Scripts Summarization.

Paket Python peringkasan teks berbahasa Bali untuk Modul 3 SatuaGen-AI (proyek Dana Indonesia Raya 2026).

| Sub-paket | Isi | Status |
|---|---|---|
| `satuasummarization.extractive` | Peringkasan ekstraktif: bobot fitur kalimat dioptimasi metaheuristik (47 algoritme dari `metaheuristicpy`) | Tersedia, beserta 47 model pretrained |
| `satuasummarization.abstractive` | Peringkasan abstraktif berbasis LLM | Dicadangkan (kosong) |

Penulis: I Made Satria Bimantara.

## Cara kerja (ekstraktif)

1. Setiap kalimat dokumen diubah menjadi 13 fitur yang dinormalisasi min-max di dalam dokumen
   (`SentenceFeatureExtractor`): panjang, posisi, rasio angka, Jaccard judul, rerata TF-ISF, kosinus ke
   sentroid dan ke judul, kemiripan antarkalimat, rasio dialog, kepadatan nomina/verba/nama diri (POS HMM
   `balinese-nlp`), dan kepadatan tokoh (SatuaNER).
2. Skor kalimat `score(s) = w · x(s)`, dengan `w ∈ [0,1]^13` hasil optimasi metaheuristik.
3. `k = ceil(rasio · n)` kalimat berskor tertinggi diambil dan disusun menurut urutan aslinya.
4. Pelatihan memaksimalkan rerata (ROUGE-1 F1 + ROUGE-2 F1)/2 pada dokumen latih, dengan `k` sama dengan jumlah
   kalimat referensi. Semua algoritme memakai anggaran evaluasi yang sama (30 agen × 100 iterasi, 3 seed).
5. Model `best` dipilih dengan 5-fold cross-validation di dalam data latih, bukan dari data uji.

## Instalasi

```bash
python -m pip install "satuasummarization[nlp]"          # inferensi (POS + SatuaNER dari balinese-nlp 2.4.2)
python -m pip install "satuasummarization[train]"        # hanya untuk melatih ulang (metaheuristicpy >= 3.2.0)
```

## Pemakaian

```python
from satuasummarization.extractive import MetaheuristicSummarizer, available_models

model = MetaheuristicSummarizer.from_pretrained("best")          # model terbaik (lihat index.json)
out = model.summarize(teks, title="I Kancil Nulungin I Kakua", ratio=0.3)
print(out["summary"])                                            # kalimat terpilih
out["sentences"]   # [{'index', 'text', 'score', 'selected'}, ...]

MetaheuristicSummarizer.from_pretrained("GWO")                   # model metaheuristik tertentu
available_models()["models"]["GWO"]                              # metrik uji model tersebut
```

Melatih dengan data sendiri:

```python
from satuasummarization.extractive import SentenceFeatureExtractor, load_corpus, train_test_split_docs
docs = load_corpus("datasets/BalineseSummarizationDatasets/extractive")   # .xlsx: sentences, extractive_summary
fx = SentenceFeatureExtractor()
for d in docs:
    d.features = fx.transform(d.sentences, d.title)
train, test = train_test_split_docs(docs, test_size=0.2, seed=42)
m = MetaheuristicSummarizer(ratio=0.4).fit(train, algorithm="PLO", n_agents=30, max_iter=100, seed=0)
m.evaluate(test)                    # ROUGE-1/2/L P/R/F1, akurasi, P/R/F1 makro kalimat
m.save("plo.json")
```

Model pretrained berupa JSON (bobot + metadata), bukan pickle. `from_pretrained` memeriksa sha256 setiap berkas
terhadap `pretrained_optimal_models/index.json` sebelum dipakai.

## Reproduksi

```bash
python scripts/extract_features.py      # 463 dokumen, 20.029 kalimat -> results/corpus_features.pkl (±30 menit, SatuaNER)
python scripts/train_pretrained.py      # 47 algoritme x 3 seed -> pretrained_optimal_models/ + results/pretrained_results.csv
python -m pytest                        # uji paket
```

## Hasil

Pemilihan model: 5-fold cross-validation di dalam 80% data latih (370 dokumen); skor CV = rerata ROUGE-1/2/L F1
pada lipatan validasi. Model `best` adalah model dengan skor CV tertinggi. Data uji (93 dokumen, 20% per sumber,
seed 42) hanya dipakai untuk pelaporan. Bobot yang dirilis dilatih pada seluruh data latih (3 seed, fitness latih
terbaik). Panjang ringkasan saat evaluasi = jumlah kalimat referensi. Sumber: `results/pretrained_results.csv`.

| # | Model | Skor CV (5-fold) | ROUGE-1 F1 uji | ROUGE-2 F1 uji | ROUGE-L F1 uji | F1 makro kalimat | Waktu latih / seed (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | APO **(best)** | 0.6719 | 0.7363 | 0.6726 | 0.6776 | 0.6246 | 23 |
| 2 | MGWO | 0.6711 | 0.7386 | 0.6769 | 0.6808 | 0.6264 | 28 |
| 3 | GWOWOA | 0.6707 | 0.7368 | 0.6759 | 0.6784 | 0.6307 | 24 |
| 4 | GWO | 0.6706 | 0.7334 | 0.6713 | 0.6752 | 0.6272 | 25 |
| 5 | SMA | 0.6704 | 0.7343 | 0.6726 | 0.6746 | 0.6201 | 20 |
| 6 | mEGO | 0.6702 | 0.7385 | 0.6788 | 0.6829 | 0.6305 | 116 |
| 7 | MGO | 0.6697 | 0.7311 | 0.6697 | 0.6744 | 0.6178 | 24 |
| 8 | EPO | 0.6695 | 0.7296 | 0.6681 | 0.6692 | 0.6165 | 27 |
| 9 | IAKMA | 0.6693 | 0.7313 | 0.6696 | 0.6729 | 0.6174 | 24 |
| 10 | DRA | 0.6692 | 0.7331 | 0.6700 | 0.6741 | 0.6204 | 207 |
| 11 | MO | 0.6689 | 0.7270 | 0.6645 | 0.6679 | 0.6110 | 23 |
| 12 | SSA | 0.6686 | 0.7243 | 0.6608 | 0.6656 | 0.6127 | 21 |
| 13 | GTO | 0.6684 | 0.7283 | 0.6653 | 0.6673 | 0.6116 | 22 |
| 14 | CFOA | 0.6684 | 0.7336 | 0.6740 | 0.6768 | 0.6249 | 24 |
| 15 | IGWO | 0.6679 | 0.7371 | 0.6775 | 0.6808 | 0.6271 | 21 |
| 16 | GA | 0.6679 | 0.7355 | 0.6723 | 0.6774 | 0.6234 | 23 |
| 17 | PSOGWO | 0.6679 | 0.7369 | 0.6758 | 0.6793 | 0.6290 | 23 |
| 18 | FLA | 0.6672 | 0.7367 | 0.6749 | 0.6792 | 0.6272 | 228 |
| 19 | EGO | 0.6667 | 0.7342 | 0.6726 | 0.6742 | 0.6242 | 134 |
| 20 | WaOA | 0.6663 | 0.7306 | 0.6693 | 0.6725 | 0.6184 | 24 |
| 21 | HGS | 0.6659 | 0.7331 | 0.6699 | 0.6744 | 0.6197 | 24 |
| 22 | FATA | 0.6657 | 0.7345 | 0.6739 | 0.6774 | 0.6258 | 24 |
| 23 | HGWODE | 0.6657 | 0.7375 | 0.6768 | 0.6797 | 0.6259 | 26 |
| 24 | RIME | 0.6656 | 0.7307 | 0.6679 | 0.6728 | 0.6200 | 24 |
| 25 | AOA | 0.6656 | 0.7368 | 0.6734 | 0.6786 | 0.6245 | 21 |
| 26 | PO | 0.6655 | 0.7416 | 0.6808 | 0.6819 | 0.6333 | 25 |
| 27 | ECO | 0.6652 | 0.7295 | 0.6665 | 0.6723 | 0.6171 | 255 |
| 28 | AO | 0.6646 | 0.7334 | 0.6730 | 0.6761 | 0.6246 | 259 |
| 29 | PFA | 0.6646 | 0.7341 | 0.6716 | 0.6754 | 0.6239 | 226 |
| 30 | SAR | 0.6643 | 0.7370 | 0.6746 | 0.6793 | 0.6282 | 257 |
| 31 | MBKA | 0.6641 | 0.7313 | 0.6661 | 0.6729 | 0.6182 | 216 |
| 32 | CFO | 0.6640 | 0.7356 | 0.6719 | 0.6743 | 0.6237 | 24 |
| 33 | RUN | 0.6638 | 0.7309 | 0.6695 | 0.6735 | 0.6184 | 260 |
| 34 | BEO | 0.6637 | 0.7333 | 0.6708 | 0.6745 | 0.6218 | 22 |
| 35 | DIO | 0.6636 | 0.7353 | 0.6755 | 0.6789 | 0.6286 | 23 |
| 36 | HO | 0.6633 | 0.7290 | 0.6683 | 0.6702 | 0.6174 | 230 |
| 37 | PSO | 0.6632 | 0.7388 | 0.6790 | 0.6824 | 0.6345 | 24 |
| 38 | KMA | 0.6630 | 0.7295 | 0.6649 | 0.6715 | 0.6159 | 25 |
| 39 | WOA | 0.6629 | 0.7256 | 0.6646 | 0.6676 | 0.6129 | 20 |
| 40 | HHO | 0.6627 | 0.7361 | 0.6781 | 0.6802 | 0.6304 | 115 |
| 41 | GCRA | 0.6627 | 0.7299 | 0.6691 | 0.6707 | 0.6185 | 256 |
| 42 | PLO | 0.6619 | 0.7307 | 0.6700 | 0.6731 | 0.6219 | 24 |
| 43 | PUMA | 0.6618 | 0.7363 | 0.6763 | 0.6810 | 0.6293 | 238 |
| 44 | HOA | 0.6608 | 0.7262 | 0.6627 | 0.6671 | 0.6105 | 227 |
| 45 | AHO | 0.6600 | 0.7289 | 0.6660 | 0.6687 | 0.6145 | 22 |
| 46 | INFO | 0.6594 | 0.7320 | 0.6678 | 0.6734 | 0.6185 | 255 |
| 47 | EEGWO | 0.6584 | 0.7331 | 0.6732 | 0.6748 | 0.6250 | 23 |
| – | Lead-k (baseline) | – | 0.6398 | 0.5501 | 0.5428 | 0.5543 | – |
| – | EqualWeights (baseline) | – | 0.7144 | 0.6464 | 0.6515 | 0.6006 | – |
| – | RandomWeights (baseline) | – | 0.7174 | 0.6509 | 0.6551 | 0.6020 | – |

Catatan: semua model metaheuristik lebih baik daripada ketiga baseline, tetapi selisih antaralgoritme kecil
(ROUGE-1 F1 uji 0.7243–0.7416; korelasi skor CV dengan ROUGE-1 uji hanya 0.21),
sehingga peringkat antaralgoritme tidak boleh ditafsirkan sebagai perbedaan yang bermakna.

Pengujian paket: `python -m pytest` → 10 tes lulus.
