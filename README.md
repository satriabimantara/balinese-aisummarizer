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
   kalimat referensi. Semua algoritme memakai anggaran evaluasi yang sama (30 agen × 100 iterasi, 3 seed;
   seed dengan fitness latih terbaik yang disimpan).

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

Data uji: 93 dokumen (20% per sumber, seed 42), panjang ringkasan = jumlah kalimat referensi. Baseline: Lead-k
(k kalimat pertama), bobot seragam (semua fitur = 1), bobot acak. Sumber: `results/pretrained_results.csv`;
kurva konvergensi per seed: `results/pretrained_convergence.json`.

| # | Model | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | F1 makro kalimat | Fitness latih (rerata ± sd, 3 seed) | Waktu latih / seed (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | PO **(best)** | 0.7416 | 0.6808 | 0.6819 | 0.6333 | 0.6872 ± 0.0015 | 57 |
| 2 | PSO | 0.7388 | 0.6790 | 0.6824 | 0.6345 | 0.6889 ± 0.0019 | 26 |
| 3 | MGWO | 0.7386 | 0.6769 | 0.6808 | 0.6264 | 0.6920 ± 0.0002 | 30 |
| 4 | mEGO | 0.7385 | 0.6788 | 0.6829 | 0.6305 | 0.6869 ± 0.0016 | 197 |
| 5 | HGWODE | 0.7375 | 0.6768 | 0.6797 | 0.6259 | 0.6932 ± 0.0007 | 39 |
| 6 | IGWO | 0.7371 | 0.6775 | 0.6808 | 0.6271 | 0.6903 ± 0.0007 | 48 |
| 7 | SAR | 0.7370 | 0.6746 | 0.6793 | 0.6282 | 0.6904 ± 0.0004 | 377 |
| 8 | PSOGWO | 0.7369 | 0.6758 | 0.6793 | 0.6290 | 0.6902 ± 0.0027 | 24 |
| 9 | AOA | 0.7368 | 0.6734 | 0.6786 | 0.6245 | 0.6907 ± 0.0012 | 38 |
| 10 | GWOWOA | 0.7368 | 0.6759 | 0.6784 | 0.6307 | 0.6901 ± 0.0012 | 38 |
| 11 | FLA | 0.7367 | 0.6749 | 0.6792 | 0.6272 | 0.6913 ± 0.0023 | 419 |
| 12 | PUMA | 0.7363 | 0.6763 | 0.6810 | 0.6293 | 0.6873 ± 0.0011 | 375 |
| 13 | APO | 0.7363 | 0.6726 | 0.6776 | 0.6246 | 0.6908 ± 0.0010 | 25 |
| 14 | HHO | 0.7361 | 0.6781 | 0.6802 | 0.6304 | 0.6850 ± 0.0008 | 150 |
| 15 | CFO | 0.7356 | 0.6719 | 0.6743 | 0.6237 | 0.6859 ± 0.0014 | 20 |
| 16 | GA | 0.7355 | 0.6723 | 0.6774 | 0.6234 | 0.6900 ± 0.0021 | 35 |
| 17 | DIO | 0.7353 | 0.6755 | 0.6789 | 0.6286 | 0.6915 ± 0.0008 | 23 |
| 18 | FATA | 0.7345 | 0.6739 | 0.6774 | 0.6258 | 0.6891 ± 0.0010 | 39 |
| 19 | SMA | 0.7343 | 0.6726 | 0.6746 | 0.6201 | 0.6824 ± 0.0013 | 32 |
| 20 | EGO | 0.7342 | 0.6726 | 0.6742 | 0.6242 | 0.6877 ± 0.0008 | 232 |
| 21 | PFA | 0.7341 | 0.6716 | 0.6754 | 0.6239 | 0.6894 ± 0.0005 | 393 |
| 22 | CFOA | 0.7336 | 0.6740 | 0.6768 | 0.6249 | 0.6877 ± 0.0016 | 22 |
| 23 | AO | 0.7334 | 0.6730 | 0.6761 | 0.6246 | 0.6892 ± 0.0003 | 310 |
| 24 | GWO | 0.7334 | 0.6713 | 0.6752 | 0.6272 | 0.6921 ± 0.0012 | 35 |
| 25 | BEO | 0.7333 | 0.6708 | 0.6745 | 0.6218 | 0.6877 ± 0.0013 | 21 |
| 26 | HGS | 0.7331 | 0.6699 | 0.6744 | 0.6197 | 0.6869 ± 0.0037 | 48 |
| 27 | EEGWO | 0.7331 | 0.6732 | 0.6748 | 0.6250 | 0.6802 ± 0.0010 | 38 |
| 28 | DRA | 0.7331 | 0.6700 | 0.6741 | 0.6204 | 0.6908 ± 0.0022 | 358 |
| 29 | INFO | 0.7320 | 0.6678 | 0.6734 | 0.6185 | 0.6909 ± 0.0011 | 298 |
| 30 | MBKA | 0.7313 | 0.6661 | 0.6729 | 0.6182 | 0.6867 ± 0.0006 | 391 |
| 31 | IAKMA | 0.7313 | 0.6696 | 0.6729 | 0.6174 | 0.6880 ± 0.0010 | 23 |
| 32 | MGO | 0.7311 | 0.6697 | 0.6744 | 0.6178 | 0.6882 ± 0.0007 | 34 |
| 33 | RUN | 0.7309 | 0.6695 | 0.6735 | 0.6184 | 0.6898 ± 0.0027 | 293 |
| 34 | RIME | 0.7307 | 0.6679 | 0.6728 | 0.6200 | 0.6922 ± 0.0005 | 47 |
| 35 | PLO | 0.7307 | 0.6700 | 0.6731 | 0.6219 | 0.6846 ± 0.0016 | 31 |
| 36 | WaOA | 0.7306 | 0.6693 | 0.6725 | 0.6184 | 0.6904 ± 0.0026 | 84 |
| 37 | GCRA | 0.7299 | 0.6691 | 0.6707 | 0.6185 | 0.6821 ± 0.0031 | 314 |
| 38 | EPO | 0.7296 | 0.6681 | 0.6692 | 0.6165 | 0.6843 ± 0.0016 | 38 |
| 39 | ECO | 0.7295 | 0.6665 | 0.6723 | 0.6171 | 0.6873 ± 0.0020 | 376 |
| 40 | KMA | 0.7295 | 0.6649 | 0.6715 | 0.6159 | 0.6854 ± 0.0011 | 54 |
| 41 | HO | 0.7290 | 0.6683 | 0.6702 | 0.6174 | 0.6840 ± 0.0001 | 456 |
| 42 | AHO | 0.7289 | 0.6660 | 0.6687 | 0.6145 | 0.6775 ± 0.0029 | 22 |
| 43 | GTO | 0.7283 | 0.6653 | 0.6673 | 0.6116 | 0.6861 ± 0.0010 | 61 |
| 44 | MO | 0.7270 | 0.6645 | 0.6679 | 0.6110 | 0.6831 ± 0.0007 | 24 |
| 45 | HOA | 0.7262 | 0.6627 | 0.6671 | 0.6105 | 0.6807 ± 0.0016 | 425 |
| 46 | WOA | 0.7256 | 0.6646 | 0.6676 | 0.6129 | 0.6825 ± 0.0009 | 48 |
| 47 | SSA | 0.7243 | 0.6608 | 0.6656 | 0.6127 | 0.6875 ± 0.0023 | 43 |
| – | RandomWeights (baseline) | 0.7174 | 0.6509 | 0.6551 | 0.6020 | – | – |
| – | EqualWeights (baseline) | 0.7144 | 0.6464 | 0.6515 | 0.6006 | – | – |
| – | Lead-k (baseline) | 0.6398 | 0.5501 | 0.5428 | 0.5543 | – | – |

Catatan: semua model metaheuristik lebih baik daripada ketiga baseline, tetapi selisih antaralgoritme kecil
(ROUGE-1 F1 0.7243–0.7416). Precision, recall, dan F1 makro kalimat bernilai sama karena
jumlah kalimat terpilih sama dengan jumlah kalimat referensi. Model `best` dipilih dari skor uji (lihat `selection_rule`
di `index.json`); untuk klaim ilmiah, pemilihan model sebaiknya memakai data validasi terpisah.

Pengujian paket: `python -m pytest` → 9 tes lulus (fitur, ROUGE vs implementasi acuan, fit/simpan/muat, validasi
masukan, integritas 47 bundel, penolakan berkas yang diubah, fitur linguistik balinese-nlp).
