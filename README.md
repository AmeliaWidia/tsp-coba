# TSP Metaheuristic: Simulated Annealing, Tabu Search, dan VNS

## Rancangan perbandingan algoritma

- Metode utama: Simulated Annealing (SA), Tabu Search (TS), dan VNS (sebagai metode tambahan).
- Satu pengaturan **Max iterasi bersama** berada di sidebar dan digunakan oleh SA, TS, serta VNS.
- Semua metode utama dimulai dari rute awal yang sama tanpa 2-opt awal.
- **Local Search + 2-opt** dijalankan terpisah sebagai baseline pembanding.
- Multi-run menampilkan hasil tiap run secara individual (seed, jarak awal/akhir, improvement, runtime, iterasi, dan evaluasi yang dilaporkan), serta ringkasan statistik.
- Operator 3-opt tidak ditawarkan pada UI karena implementasi awalnya belum setara dengan definisi 3-opt standar; operator yang tersedia untuk SA/TS adalah Swap dan 2-opt.

Catatan: jumlah iterasi yang sama tidak selalu berarti jumlah evaluasi kandidat yang sama. SA mengevaluasi satu kandidat per iterasi, sedangkan Tabu Search mengevaluasi sejumlah kandidat sesuai ukuran neighborhood.


Aplikasi berbasis **Streamlit** untuk menyelesaikan permasalahan *Traveling Salesman Problem* (TSP) menggunakan tiga algoritma metaheuristik: **Simulated Annealing (SA), Tabu Search (TS), dan Variable Neighborhood Search (VNS)**.

Aplikasi ini mendukung pembentukan solusi awal, optimasi rute, visualisasi, analisis konvergensi, perbandingan kinerja algoritma, dan pengujian berulang untuk mengevaluasi kualitas solusi serta waktu komputasi berdasarkan hasil aktual.

## Fitur Utama

- **Input Data:** menambahkan kota melalui klik kanvas X–Y, membuat koordinat acak, mengedit tabel secara manual, mengunggah CSV/Excel, dan menggunakan dataset contoh.
- **Perhitungan Jarak:** memilih metrik Euclidean atau Manhattan.
- **Pembentukan Solusi Awal:** menggunakan Nearest Neighbor (NN), Nearest Insertion (NI), Cheapest Insertion (CI), Farthest Insertion (FI), Arbitrary Insertion (AI), atau rute manual.
- **Pencarian Lokal:** menerapkan 2-opt pada solusi awal atau melakukan perbaikan rute setelah optimasi.
- **Simulated Annealing:** menerima solusi berdasarkan perubahan jarak, probabilitas Metropolis, dan penurunan suhu.
- **Tabu Search:** mengeksplorasi solusi tetangga menggunakan tabu list FIFO dan kriteria aspirasi.
- **Variable Neighborhood Search:** mengganti struktur lingkungan pencarian menggunakan Swap, Insertion, dan 2-opt, dilengkapi mekanisme pengguncangan solusi dan opsi pencarian lokal.
- **Perbandingan Algoritma:** membandingkan jarak terbaik, waktu komputasi, jumlah iterasi, dan struktur rute SA, TS, serta VNS.
- **Analisis Konvergensi:** menampilkan perubahan jarak terbaik berdasarkan riwayat optimasi aktual.
- **Analisis Hasil:** menyajikan kualitas solusi, peningkatan jarak, waktu komputasi, pembaruan solusi terbaik, dan kesamaan sisi rute dalam enam kartu informasi.
- **Pengujian Berulang:** mengevaluasi rata-rata, simpangan baku, dan konsistensi hasil antarrun.
- **Ekspor Hasil:** menyimpan visualisasi rute, perbandingan rute, grafik konvergensi, detail rute, log iterasi, dan matriks jarak.

## Struktur Proyek

```text
TSP-Metaheuristic/
├── app/
│   └── app.py              # Antarmuka dan alur aplikasi Streamlit
├── src/
│   └── tsp_core.py         # Perhitungan jarak dan algoritma optimasi
├── data/
│   └── data1.xlsx          # Dataset contoh
├── docs/                   # Dokumentasi proyek
├── .streamlit/
│   └── config.toml         # Konfigurasi tampilan aplikasi
├── requirements.txt        # Daftar pustaka yang dibutuhkan
└── README.md               # Dokumentasi utama
```

### Komponen Utama

- `app/app.py`: mengatur input data, konfigurasi algoritma, eksekusi optimasi, visualisasi, perbandingan hasil, dan analisis.
- `src/tsp_core.py`: berisi fungsi perhitungan jarak, metode pembentukan solusi awal, pencarian lokal, SA, TS, dan VNS.
- `data/data1.xlsx`: dataset contoh pada lembar kerja `Sheet7`, yang mencakup Case I (25 kota), Case II (40 kota), dan Case III (68 kota).

## Teknologi yang Digunakan

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Plotly
- openpyxl

## Instalasi dan Menjalankan Aplikasi

Pastikan Python telah terpasang pada komputer.

**1. Buka terminal pada folder proyek**

```powershell
cd "path\ke\TSP-Metaheuristic"
```

**2. Buat dan aktifkan lingkungan virtual**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika lingkungan virtual sudah tersedia, cukup aktifkan tanpa membuatnya kembali.

**3. Pasang pustaka yang dibutuhkan**

```powershell
python -m pip install -r requirements.txt
```

**4. Jalankan aplikasi**

```powershell
python -m streamlit run app/app.py
```

Buka alamat lokal yang ditampilkan pada terminal, biasanya:

`http://localhost:8501`

## Pengaturan dan Metode Optimasi

### 1. Data Kota dan Solusi Awal

Data kota dapat dimasukkan melalui kanvas koordinat X–Y, pembuatan data acak, tabel manual, unggahan CSV/Excel, atau dataset contoh.

Solusi awal dapat dibentuk menggunakan heuristik NN, NI, CI, FI, dan AI, atau ditentukan secara manual. Rute manual harus memuat seluruh kota tepat satu kali.

Aplikasi mendukung maksimal 200 kota. Untuk dataset contoh, pengguna dapat memilih kasus dan jumlah kota yang akan digunakan.

### 2. Metrik Jarak dan Pencarian Lokal

Aplikasi menyediakan dua metrik untuk menghitung jarak antar kota.

**Euclidean Distance**

Menghitung jarak garis lurus antara dua kota berdasarkan koordinat X dan Y.

$$
d(i,j)=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}
$$

**Manhattan Distance**

Menghitung jarak berdasarkan jumlah selisih absolut koordinat X dan Y.

$$
d(i,j)=|x_i-x_j|+|y_i-y_j|
$$

Pencarian lokal menggunakan operator **2-opt** untuk membalik urutan segmen rute. Strategi perbaikan yang tersedia adalah *Best Improvement*, yang memilih perbaikan terbaik pada iterasi pencarian, dan *First Improvement*, yang menerima perbaikan pertama yang ditemukan.

### 3. Simulated Annealing (SA)

SA menerima solusi yang memperpendek rute dan dapat menerima solusi yang lebih buruk berdasarkan probabilitas Metropolis agar pencarian tidak mudah terjebak pada optimum lokal.

Probabilitas penerimaan:

```text
P(accept) = 1                    jika ΔE <= 0
P(accept) = exp(-ΔE / T)         jika ΔE > 0
```

Penurunan suhu:

```text
T_next = alpha * T
```

Parameter yang tersedia meliputi suhu awal, laju pendinginan, suhu minimum, jumlah iterasi per suhu, iterasi maksimum, dan operator lingkungan pencarian.

### 4. Tabu Search (TS)

TS mengevaluasi solusi tetangga dengan menggunakan tabu list berukuran terbatas untuk mengurangi pengulangan langkah pencarian. Daftar tabu menggunakan mekanisme FIFO, sedangkan kriteria aspirasi memungkinkan langkah tabu diterima apabila menghasilkan solusi yang lebih baik daripada solusi terbaik sebelumnya.

Parameter utama meliputi iterasi maksimum, ukuran tabu list, ukuran lingkungan pencarian, operator pencarian, dan status kriteria aspirasi.

### 5. Variable Neighborhood Search (VNS)

VNS mengeksplorasi ruang solusi dengan mengganti struktur lingkungan pencarian secara sistematis.

Struktur lingkungan yang didukung:

- **Swap:** menukar posisi dua kota.
- **Insertion:** memindahkan satu kota ke posisi lain dalam rute.
- **2-opt:** membalik urutan suatu segmen rute.

Alur VNS dimulai dari solusi awal, melakukan pengguncangan (*shaking*), menjalankan pencarian lokal jika diaktifkan, lalu mengevaluasi kandidat berdasarkan total jarak. Jika kandidat lebih baik, solusi diperbarui dan pencarian kembali ke lingkungan pertama. Jika tidak, pencarian berlanjut ke lingkungan berikutnya.

Parameter VNS meliputi:

- Iterasi maksimum.
- Jumlah maksimum lingkungan (`k_max`).
- Pilihan struktur lingkungan.
- Metode pengguncangan.
- Opsi pencarian lokal 2-opt.
- *Random seed*.

VNS menggunakan metrik jarak yang dipilih pengguna serta menyimpan solusi terbaik, waktu komputasi, jumlah iterasi, dan riwayat konvergensi.

### 7. Pengujian Berulang dan Random Seed

Fitur pengujian berulang menjalankan SA, TS, dan VNS beberapa kali untuk mengukur konsistensi hasil. Tabel detail menampilkan Run 1, Run 2, dan seterusnya secara individual.

Statistik yang dapat dibandingkan meliputi:

- Jarak terbaik dan terburuk.
- Rata-rata jarak.
- Simpangan baku.
- Rata-rata waktu komputasi.

Penggunaan *random seed* memungkinkan eksperimen diulang dengan data dan konfigurasi yang sama. Waktu komputasi tetap dapat berbeda karena kondisi perangkat.

## Alur Penggunaan

1. **Masukkan data kota** dan periksa koordinatnya.
2. **Atur konfigurasi** jarak, solusi awal, pencarian lokal, SA, TS, VNS, dan parameter opsional. Batas iterasi SA dan TS memakai satu nilai bersama; SA berhenti ketika suhu minimum tercapai atau batas iterasi bersama habis.
3. **Periksa ringkasan konfigurasi** sebelum menjalankan optimasi.
4. **Tekan RUN OPTIMIZATION** untuk menjalankan algoritma.
5. **Evaluasi hasil** melalui ringkasan, visualisasi rute, perbandingan, konvergensi, log, dan analisis.
6. **Ekspor hasil** sesuai kebutuhan eksperimen atau dokumentasi penelitian.

## Hasil dan Visualisasi

| Bagian | Informasi |
|---|---|
| Ringkasan | Jarak awal, jarak terbaik, peningkatan, waktu komputasi, dan jumlah iterasi. |
| Rute | Visualisasi rute awal dan hasil optimasi. |
| Perbandingan | Kesamaan sisi rute dan perbedaan jarak antarpasangan algoritma. |
| Konvergensi | Grafik jarak terbaik berdasarkan riwayat iterasi aktual. |
| Detail & Log | Detail perjalanan antarkota, log iterasi, dan matriks jarak. |
| Analisis | Enam kartu informasi yang membahas kualitas solusi, runtime, perbaikan, dan struktur rute. |
| Tabu Explorer | Menelusuri riwayat Tabu Search untuk dataset maksimal 12 kota. |
| Pengujian Berulang | Statistik eksperimen jika fitur multi-run diaktifkan. |

### Ekspor

Hasil yang tersedia untuk diekspor meliputi:

- **PNG:** visualisasi rute, perbandingan rute, dan grafik konvergensi.
- **CSV:** detail rute, log iterasi, dan matriks jarak.

## Catatan Implementasi

- Rute TSP merupakan perjalanan tertutup: kota terakhir harus kembali ke kota awal.
- Setiap kota harus dikunjungi tepat satu kali.
- SA, TS, dan VNS menggunakan data kota, metrik jarak, serta solusi awal yang sama dalam satu eksekusi perbandingan.
- Jika pencarian lokal setelah optimasi diaktifkan, hasil akhir turut memperhitungkan perbaikan 2-opt.
- Jarak, peningkatan, runtime, iterasi, konvergensi, dan kesamaan sisi dihitung dari eksekusi algoritma.
- Tidak ada algoritma yang dijamin selalu menghasilkan rute terbaik untuk setiap dataset. Kualitas solusi dipengaruhi data, parameter, seed, dan proses pencarian.

## Pemecahan Masalah

**Streamlit tidak dikenali**

```powershell
python -m streamlit run app/app.py
```

**Pustaka belum terpasang**

```powershell
python -m pip install -r requirements.txt
```

**Dataset contoh tidak ditemukan**

Pastikan file `data/data1.xlsx` tersedia dan memiliki lembar kerja `Sheet7`.

**Hasil tidak sesuai dengan konfigurasi terbaru**

Jalankan kembali **RUN OPTIMIZATION** setelah mengubah data atau parameter agar hasil mencerminkan konfigurasi yang sedang digunakan.

---

**Tujuan Proyek**

Proyek ini dikembangkan sebagai sarana pembelajaran dan eksperimen optimasi kombinatorial, dengan fokus pada perbandingan kualitas rute, perilaku konvergensi, waktu komputasi, dan konsistensi hasil dari Simulated Annealing, Tabu Search, dan Variable Neighborhood Search.