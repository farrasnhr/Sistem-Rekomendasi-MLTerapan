# Laporan Proyek Machine Learning - Farras Nur Haidar Ramadhan
## Domain Proyek
### Latar Belakang
Saat ini, layanan video berlangganan di internet, atau yang dikenal sebagai layanan *streaming*, semakin populer dan mencakup berbagai *platform* besar seperti Netflix, Disney+, Iflix, Amazon Prime, dan lainnya. *Platform-platform* ini menawarkan ribuan hingga puluhan ribu judul film, serial, dokumenter, dan konten *original* yang dirilis secara rutin. Layanan video berlangganan di internet telah mengalami peningkatan drastis dalam total waktu tonton selama sepuluh tahun terakhir. Pada tahun 2017, pengguna Netflix secara kolektif menonton konten selama 140 juta jam per hari, menghasilkan pendapatan sebesar 11 miliar dolar Amerika[^1].<br>
Per September 2024, IMDb melaporkan memiliki lebih dari 20 juta judul dalam basis datanya. Jumlah ini mencakup berbagai jenis konten, termasuk film, serial televisi, video musik, podcast, dan lainnya. Angka ini terus bertambah seiring dengan penambahan judul baru ke dalam basis data IMDb[^2].<br>

Namun, dengan lebih dari 20 juta judul yang tercatat di IMDb per September 2024, pengguna sering kali kesulitan menemukan konten yang relevan di tengah banyaknya pilihan yang tersedia. Inilah yang membuat sistem rekomendasi menjadi sangat penting dalam industri ini. Sistem rekomendasi dapat membantu menyaring dan menyesuaikan konten yang sesuai dengan preferensi pengguna, menciptakan pengalaman menonton yang lebih personal dan efisien. Seperti pada kutipan dari artikel dari Gomez dan Hunt yang berbunyi "**Pengguna akan menemukan sesuatu yang menarik atau risiko pengguna meninggalkan layanan kami akan meningkat secara signifikan.**"[^3] Kalimat tersebut menggambarkan pentingnya rekomendasi yang relevan dalam layanan. Jika pengguna tidak menemukan konten yang menarik, ada kemungkinan besar mereka akan merasa frustasi dan berhenti menggunakan layanan tersebut. Oleh karena itu, sistem rekomendasi yang efektif sangat penting untuk mempertahankan keterlibatan pengguna dan mengurangi risiko kehilangan pelanggan.<br>

Berdasarkan latar belakang di atas, dapat dirancang sebuah model machine learning untuk memberikan rekomendasi yang relevan kepada pengguna. Model ini bertujuan untuk memudahkan pengguna menemukan film yang sesuai dengan preferensi mereka tanpa harus melakukan pencarian manual.

## Business Understanding
Dari penjelasan latar belakang di atas, dapat dibuat rumusan masalah sebagai berikut:

### Problem Statements
- Bagaimana menyiapkan data yang diperlukan untuk membuat model sistem rekomendasi?
- Bagaimana cara membuat model untuk merekomendasikan film?
  
### Goals
Berdasarkan rumusan masalah sebelumnya, dapat dibuatkan tujuan laporan sebagai berikut
- Melakukan tahapan persiapan data, agar data yang sudah disiapkan dapat dimasukkan ke dalam model,
- Membuat model sistem rekomendasi film untuk pengguna.
  
### Solution Statemnent
Berdasarkan dari tujuan, didapatkan beberapa solusi untuk menjawab rumusan masalah sebagai berikut:
1. Persiapan Data
- Melakukan *Data preprocessing* terpisah pada dua teknik sistem rekomendasi
- CBF<br>
  Dengan melakukan merubah tipe data dan menggabungkan fitur dari berkas lain dengan mencocokkan `id` pada berkas, merubah data teks menjadi representasi vektor dengan TF-IDF.<br>
- Collaborative Filtering<br>
  Dengan melakukan merubah tipe data dan menggabungkan fitur dari berkas lain dengan mencocokkan `id`, dan melakukan pemetaan, konversi tipe data, mapping, dan pembagian data. <br>
2. Modeling
- Membangun model dengan dua teknik sistem rekomendasi, yaitu:
  1. Content-Based Filtering,
     - Dengan meggunakan `cosine_similarity`.
  2. Collaborative Filtering.
     - Dengan menggunakan *library* dari **Keras**
3. Melakukan pengujian model sistem rekomendasi setelah proses *Modeling*.
  
## *Library* Untuk Proyek
Menyiapkan beberapa *library* guna menunjang pengerjaan proyek. Berikut *library* yang akan digunakan pada proyek ini:
```python
import os
import ast
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras import layers
```

## Data Understanding
Dataset yang digunakan untuk proyek ini adalah [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data) yang diambil dari laman Kaggle. Dataset tersebut memiliki tujuh berkas dengan format `csv` berukuran total 228 MB.

### Informasi Dataset
Dari masing-masing berkas csv memiliki informasi sebagai berikut:
1. **credits**<br>
  **RangeIndex**: 45476 entries<br>
  
    | # | Column     | Non-Null Count | Dtype   |
    |---|------------|----------------|---------|
    | 0 | cast       | 45476 non-null | object  |
    | 1 | crew       | 45476 non-null | object  |
    | 2 | id         | 45476 non-null | int64   |
   
   Tabel di atas menunjukkan terdapat 45476 data dengan tipe data `int64` sebanyak satu kolom, dan empat lainnya bertipe data `object`. Dengan uraian setiap fitur sebagai berikut:<br>
   - `cast`: Berisi informasi tentang pemeran film,
   - `crew`: Berisi informasi tentang kru film, termasuk sutradara, penulis naskah, dan lain-lain,
   - `id`: Berisi ID *unique* untuk setiap film yang sesuai dengan id di file movies_metadata.csv.
 
2. **keywords**<br>
   **RangeIndex**: 46419 entries<br>
   
    | # | Column   | Non-Null Count | Dtype   |
    |---|----------|----------------|---------|
    | 0 | id       | 46419 non-null | int64   |
    | 1 | keywords | 46419 non-null | object  |

    Tabel di atas menunjukkan terdapat 46419 data dengan tipe data `int64` sebanyak satu kolom, dan satu bertipe data `object`. Dengan uraian setiap fitur sebagai berikut:<br>
    - `id`: Berisi ID *unique* untuk setiap film yang sesuai dengan `id` di file `movies_metadata.csv`,
    - `keywords`: Berisi daftar kata kunci (keywords) yang relevan dengan film.

3. **links**<br>
   **RangeIndex**: 45843 entries<br>
   
    | # | Column  | Non-Null Count | Dtype   |
    |---|---------|----------------|---------|
    | 0 | movieId | 45843 non-null | int64   |
    | 1 | imdbId  | 45843 non-null | int64   |
    | 2 | tmdbId  | 45624 non-null | float64 |

    Tabel di atas menunjukkan terdapat 45843 data dengan tipe data `float64` sebanyak satu kolom, dan dua lainnya bertipe data `int64`. Dengan uraian setiap fitur sebagai berikut:<br>
      - `movieId`: Berisi ID *unique* film yang sesuai dengan `movieId` di file `ratings.csv`,
      - `imdbId`: Berisi ID film di IMDb, digunakan untuk menghubungkan data dengan sumber luar, seperti situs IMDb,
      - `tmdbId`: Berisi ID film di TMDb (The Movie Database), juga digunakan untuk menghubungkan data dengan TMDb.
    
4. **links_small**<br>
   **RangeIndex**: 9125 entries<br>
    | # | Column  | Non-Null Count | Dtype   |
    |---|---------|----------------|---------|
    | 0 | movieId | 9125 non-null  | int64   |
    | 1 | imdbId  | 9125 non-null  | int64   |
    | 2 | tmdbId  | 9112 non-null  | float64 |

   Tabel di atas menunjukkan terdapat 9125 data dengan tipe data `float64` sebanyak satu kolom, dan dua lainnya bertipe data `int64`. Dengan uraian setiap fitur sebagai berikut:<br>
     - `movieId`: Berisi ID *unique* film yang sesuai dengan `movieId` di file `ratings.csv`,
     - `imdbId`: Berisi ID film di IMDb, digunakan untuk menghubungkan data dengan sumber luar, seperti situs IMDb,
     - `tmdbId`: Berisi ID film di TMDb (The Movie Database), juga digunakan untuk menghubungkan data dengan TMDb.
   
5. **movies_metadata**<br>
   **RangeIndex**: 45466 entries<br>
    | # | Column                 | Non-Null Count | Dtype   |
    |---|-------------------------|----------------|---------|
    | 0 | adult                   | 45466 non-null | object  |
    | 1 | belongs_to_collection   | 4494 non-null  | object  |
    | 2 | budget                  | 45466 non-null | object  |
    | 3 | genres                  | 45466 non-null | object  |
    | 4 | homepage                | 7782 non-null  | object  |
    | 5 | id                      | 45466 non-null | int64   |
    | 6 | imdb_id                 | 45449 non-null | object  |
    | 7 | original_language       | 45455 non-null | object  |
    | 8 | original_title          | 45466 non-null | object  |
    | 9 | overview                | 45412 non-null | object  |
    | 10 | popularity             | 45461 non-null | object  |
    | 11 | poster_path            | 45080 non-null | object  |
    | 12 | production_companies   | 45463 non-null | object  |
    | 13 | production_countries   | 45463 non-null | object  |
    | 14 | release_date           | 45379 non-null | object  |
    | 15 | revenue                | 45466 non-null | float64 |
    | 16 | runtime                | 45204 non-null | float64 |
    | 17 | spoken_languages       | 45460 non-null | object  |
    | 18 | status                 | 45460 non-null | object  |
    | 19 | tagline                | 20412 non-null | object  |
    | 20 | title                  | 45460 non-null | object  |
    | 21 | video                  | 45460 non-null | object  |
    | 22 | vote_average           | 45460 non-null | float64 |
    | 23 | vote_count             | 45460 non-null | float64 |

    Tabel di atas menunjukkan terdapat 45466 data dengan tipe data `float64` sebanyak empat kolom, dan 20 lainnya bertipe data `int64`. Dengan uraian setiap fitur sebagai berikut:<br>
      - `adult`: Berisi apakah film ini untuk dewasa atau tidak,
      - `belongs_to_collection`: Berisi informasi tentang koleksi film,
      - `budget`: Berisi anggaran pembuatan film,
      - `genres`: Berisi daftar genre film,
      - `homepage`: Berisi URL situs resmi film,
      - `id`: Berisi ID *unique* untuk setiap film, digunakan untuk menghubungkan dengan file lain,
      - `imdb_id`: Berisi ID *unique* film di IMDb,
      - `original_language`: Berisi bahasa asli film,
      - `original_title`: Berisi judul asli film,
      - `overview`: Berisi sinopsis atau deskripsi singkat tentang film,
      - `popularity`: Berisi skor popularitas film,
      - `poster_path`: Berisi *path* atau URL untuk poster film,
      - `production_companies`: Berisi daftar perusahaan produksi yang terlibat dalam pembuatan film,
      - `production_countries`: Berisi daftar negara tempat produksi film,
      - `release_date`: Berisi tanggal rilis film,
      - `revenue`: Berisi pendapatan yang dihasilkan oleh film,
      - `runtime`: Berisi durasi film dalam menit,
      - `spoken_languages`: Berisi daftar bahasa yang digunakan dalam film,
      - `status`: Berisi status rilis film,
      - `tagline`: Berisi *tagline* atau slogan film,
      - `title`: Berisi judul film,
      - `video`: Berisi apakah film ini memiliki video terkait,
      - `vote_average`: Berisi rata-rata *rating* yang diberikan oleh pengguna,
      - `vote_count`: Berisi jumlah total *vote* atau *rating* yang diberikan oleh pengguna.

6. **ratings**<br>
   **RangeIndex**: 26024289 entries<br>
    | # | Column    | Dtype   |
    |---|-----------|---------|
    | 0 | userId    | int64   |
    | 1 | movieId   | int64   |
    | 2 | rating    | float64 |
    | 3 | timestamp | int64   |

    Tabel di atas menunjukkan terdapat 26024289 data dengan tipe data `float64` sebanyak satu kolom, dan tiga lainnya bertipe data `int64`. Dengan uraian setiap fitur sebagai berikut:<br>
      - `userId`: Berisi ID *unique* untuk setiap pengguna,
      - `movieId`: Berisi ID *unique* untuk setiap film yang sesuai dengan `movieId` di `links.csv`,
      - `rating`: Berisi rating yang diberikan oleh pengguna untuk film tersebut,
      - `timestamp`: Berisi waktu saat *rating* diberikan.
    
7. **ratings_small**<br>
   **RangeIndex**: 100004 entries<br>
    | # | Column    | Non-Null Count | Dtype   |
    |---|-----------|----------------|---------|
    | 0 | userId    | 100004 non-null | int64   |
    | 1 | movieId   | 100004 non-null | int64   |
    | 2 | rating    | 100004 non-null | float64 |
    | 3 | timestamp | 100004 non-null | int64   |
   
    Tabel di atas menunjukkan terdapat 100004 data dengan tipe data `float64` sebanyak satu kolom, dan tiga lainnya bertipe data `int64`. Dengan uraian setiap fitur sebagai berikut:<br>
      - `userId`: Berisi ID *unique* untuk setiap pengguna,
      - `movieId`: Berisi ID *unique* untuk setiap film yang sesuai dengan `movieId` di `links.csv`,
      - `rating`: Berisi *rating* yang diberikan oleh pengguna untuk film tersebut,
      - `timestamp`: Berisi waktu saat *rating* diberikan.

## EDA
### credits
1. **Missing Values**<br>
   Dimulai dengan melihat *missing values* pada berkas `credits`.<br>

| Column | Missing Values |
|--------|----------------|
| cast   | 0              |
| crew   | 0              |
| id     | 0              |
<br>

Tabel diatas menunjukkan tidak adanya *missing values* pada berkas `credits`.<br>

<br>

2. **cast *unique***<br>

- **Jumlah Cast *Unique*:** 202747
- **Contoh Nama Cast *Unique*:**
  - Adrianne Palicki
  - Helma Vandenberg
  - Anne Carlisle
  - Enzo Marcelli
  - Guillermo Battaglia
  - Adriano Celentano
  - Ina Sofie Brodahl
  - Andre Lindveldt
  - Arron Shiver
  - Keir Knight

Daftar diatas menunjukkan `cast` *unique* memiliki sebanyak 202747 `cast`.<br>

3. **sutradara / *director* *unique***.<br>

- **Jumlah Sutradara Unik:** 19740
- **Beberapa Nama Sutradara:**
  - Josef Hader
  - Júlíus Kemp
  - James W. Griffiths
  - Sue Bourne
  - Walter Doniger
  - Adriano Celentano
  - Chris Alexander
  - John Cassavetes
  - Alan Cumming
  - Patrice Toye

Daftar diatas menunjukkan `crew` khususnya `sutradara / director` yang *unique* memiliki sebanyak 19740 `sutradara / director`.<br>

4. **`cast` yang paling banyak muncul di film**.<br>

- Cast yang Paling Sering Muncul.<br>

| Nama Cast          | Frekuensi |
|--------------------|-----------|
| Bess Flowers       | 241 kali  |
| Christopher Lee    | 148 kali  |
| John Wayne         | 125 kali  |
| Samuel L. Jackson  | 123 kali  |
| Gérard Depardieu   | 110 kali  |
| Michael Caine      | 110 kali  |
| Donald Sutherland  | 109 kali  |
| John Carradine     | 109 kali  |
| Jackie Chan        | 108 kali  |
| Frank Welker       | 107 kali  |

- Cast yang Paling Jarang Muncul (Muncul Sekali).<br>

| Nama Cast           | Frekuensi |
|---------------------|-----------|
| Brandon Obray       | 1 kali    |
| Cyrus Thiedeke      | 1 kali    |
| Gary Joseph Thorup  | 1 kali    |
| Leonard Zola        | 1 kali    |
| Robyn Driscoll      | 1 kali    |
| Sarah Gilson        | 1 kali    |
| Florica Vlad        | 1 kali    |
| June Lion           | 1 kali    |
| Brenda Lockmuller   | 1 kali    |
| Brad Baldridge      | 1 kali    |

5. **movieId dengan `cast` jumlah terbanyak dan sedikit**.<br>

| **Deskripsi**                  | **Movie ID** | **Jumlah Cast** |
|--------------------------------|--------------|-----------------|
| Film dengan cast terbanyak     | 2897         | 313             |
| Film dengan cast tersedikit    | 124639       | 0               |

6. **movieId dengan `cast` jumlah terbanyak dan sedikit**.<br>

| **Deskripsi**                | **Movie ID** | **Jumlah Crew** |
|------------------------------|--------------|-----------------|
| Film dengan crew terbanyak   | 135397       | 435             |
| Film dengan crew tersedikit  | 56088        | 0               |

7. **Melihat distribusi jumlah cast per film**.<br>
   ![Distribusi Jumlah Cast per Film](https://github.com/user-attachments/assets/045bd69c-68b0-42df-adb9-58c7610e9611)

Gambar di atas menunjukkan bahwa sebagian besar film memiliki jumlah cast yang rendah, dengan puncak di sekitar 10-20 cast per film, dan hanya sedikit film yang memiliki cast dalam jumlah besar hingga lebih dari 100 orang, menandakan distribusi yang positif skewed dengan mayoritas film berada di kisaran 0-50 cast.

8. **Melihat distribusi jumlah crew per film**.<br>
   ![Distribusi Jumlah Crew per Film](https://github.com/user-attachments/assets/f80229e6-e0f6-45e9-a1e8-ffa0a7d468ba)

Gambar di atas menunjukkan bahwa mayoritas film memiliki jumlah crew yang rendah, terutama dalam rentang 0-50 anggota, dengan beberapa pengecualian yang mencapai lebih dari 100 hingga 400 anggota, memperlihatkan distribusi yang sangat positif skewed dengan sebagian besar film dikerjakan oleh tim yang relatif kecil.

9. **Melihat Top 10 `Cast` yang paling sering muncul**.<br>
   ![Top 10 Cast yang Paling Sering Muncul](https://github.com/user-attachments/assets/87309ec6-f3d1-46a8-8600-0f8270e0a27c)

Gambar di atas menunjukkan 10 cast teratas yang paling sering muncul di film. 

10. **Melihat Top 10 sutradara dengan film terbanyak**.<br>
    ![Top 10 Sutradara dengan Film Terbanyak](https://github.com/user-attachments/assets/13812b27-bf6c-4495-8152-76ef6b36f253)

Gambar di atas menunjukkan 10 sutradara dengan film terbanyak.

### keywords
1. **Missing Values**<br>
   Dimulai dengan melihat *missing values* pada berkas `keywords`.<br>

| Column   | Missing Values |
|----------|----------------|
| id       | 0              |
| keywords | 0              |

<br>

Tabel di atas menunjukkan tidak adanya *missing values* pada berkas `keywords`.<br>

2. **Melihat jumlah keywords yang *unique***.<br>

   **Jumlah Keywords Unique:** 19956  
   **Contoh Keywords Unique:**
      - catfishing
      - snooker
      - reference to elvis presley
      - braggart
      - onsen
      - prussia
      - fetishism
      - tour bus
      - klingon
      - injection

3. **Distribusi jumlah keywords per film**.
   ![Distribusi Jumlah Keywords per Film](https://github.com/user-attachments/assets/b587dcec-7cd2-4c2c-9c0a-66d7a546d9a1)

Gambar di atas menunjukkan bahwa kebanyakan film hanya memiliki sedikit kata kunci, biasanya antara 0 hingga 5. Hanya sedikit sekali film yang punya banyak kata kunci, dan semakin banyak kata kunci, semakin sedikit jumlah filmnya.

4. **Top 10 keywords**.
   ![Top 10 Keyword](https://github.com/user-attachments/assets/6fe4a17b-afac-4c64-a750-c0813aed4e5d)

Gambar di atas menunjukkan sepuluh kata kunci yang paling sering digunakan dalam film, dengan "woman director" sebagai kata kunci yang paling banyak muncul, diikuti oleh "independent film" dan "murder." Kata kunci ini mengindikasikan tema atau elemen yang sering diangkat dalam berbagai film, seperti sutradara perempuan, film independen, dan tema pembunuhan, yang semuanya memiliki frekuensi tinggi dibandingkan dengan kata kunci lain dalam daftar.

### links
1. **Missing Values**<br>
   Dimulai dengan melihat *missing values* pada berkas `links`.<br>

| Column  | Missing Values |
|---------|----------------|
| movieId | 0              |
| imdbId  | 0              |
| tmdbId  | 219            |

<br>
Tabel diatas menunjukkan adanya *missing values* pada fitur `tmdbId` sebanyak 219 data.<br>
<br>

2. **Id**.<br>

| Column  | Unique       |
|---------|--------------|
| movieId | 45843        |
| imdbId  | 45843        |
| tmdbId  | 45594        |

Tabel di atas menunjukkan Id *unique* sesuai dengan panjang baris pada berkas `links` yaitu 45843 baris, sementara pada tmdbId terdapat *missing values* seperti yang diinfokan pada *missing values*.<br>

3. **Distribusi `imdbId`**<br>

   ![Distribusi imdbId](https://github.com/user-attachments/assets/9541de78-4273-4e7a-9cdb-8e3fec19989b)

   Gambar di atas menunjukkan bahwa sebagian besar film memiliki nilai imdbId yang relatif kecil, dengan sedikit film yang memiliki nilai imdbId yang lebih tinggi, yang mengindikasikan distribusi yang sangat tidak merata di mana mayoritas film berada di kisaran rendah.

5. **Distribusi `tmdbId`**<br>

   ![Distribusi tmdbId](https://github.com/user-attachments/assets/73615ec9-e12e-4517-9777-af67c3f4a99b)

   Gambar di atas menunjukka pola serupa untuk tmdbId, dengan banyak film yang memiliki nilai tmdbId rendah dan hanya sedikit film yang memiliki nilai tmdbId tinggi, yang menunjukkan distribusi yang juga tidak merata di antara film-film tersebut.

   
### links_small
1. **Missing Values**<br>
   Dimulai dengan melihat *missing values* pada berkas `links_small`.<br>

| Column  | Missing Values |
|---------|----------------|
| movieId | 0              |
| imdbId  | 0              |
| tmdbId  | 13             |
<br>
Tabel diatas menunjukkan tidak adanya *missing values* pada berkas `credits`.<br>

2. **Id *unique***.<br>

  | Column  | Unique       |
  |---------|--------------|
  | movieId | 9125         |
  | imdbId  | 9125         |
  | tmdbId  | 9112         |

Tabel di atas menunjukkan Id *unique* sesuai dengan panjang baris pada berkas `links` yaitu 9125 baris, sementara pada tmdbId terdapat *missing values* seperti yang diinfokan pada *missing values*.<br>

3. **Distribusi `imdbId`**<br>

   ![Distribusi imdbId](https://github.com/user-attachments/assets/2441609b-6d69-441f-9904-3cf6136c9bfd)

  Gambar di atas menunjukkan distribusi imdbId, kita melihat bahwa sebagian besar film memiliki nilai imdbId yang rendah, dengan frekuensi tertinggi berkumpul di bagian kiri grafik.
  
4. **Distribusi `tmdbId`**<br>
   
   ![Distribusi tmdbId](https://github.com/user-attachments/assets/b29c6e5e-af2a-43f9-86b9-8441c24c2491)

   Gambar di atas menunjukkan distribusi tmdbId, pola yang mirip terlihat, di mana mayoritas tmdbId berjumlah rendah dan beberapa jumlah lebih tinggi muncul semakin jarang.

   
### movie_metadata
1. **Missing Values**<br>
   Dimulai dengan melihat *missing values* pada berkas `movies_metadata`.<br>

| Column                 | Missing Values |
|------------------------|----------------|
| adult                  | 0              |
| belongs_to_collection  | 40972          |
| budget                 | 0              |
| genres                 | 0              |
| homepage               | 37684          |
| id                     | 0              |
| imdb_id                | 17             |
| original_language      | 11             |
| original_title         | 0              |
| overview               | 954            |
| popularity             | 5              |
| poster_path            | 386            |
| production_companies   | 3              |
| production_countries   | 3              |
| release_date           | 87             |
| revenue                | 6              |
| runtime                | 263            |
| spoken_languages       | 6              |
| status                 | 87             |
| tagline                | 25054          |
| title                  | 6              |
| video                  | 6              |
| vote_average           | 6              |
| vote_count             | 6              |

<br>

Tabel di atas menunjukkan adanya *missing values* pada beberapa fitur, yaitu `belongs_to_collection` (40972), `homepage` (37684), `imdb_id` (17), `original_language` (11), `overview` (954), `popularity` (5), `poster_path` (386), `production_companies` (3), `production_countries` (3), `release_date` (87), `revenue` (6), `runtime` (263), `spoken_languages` (6), `status` (87), `tagline` (25054), `title` (6), `video` (6), `vote_average` (6), dan `vote_count` (6).<br>

2. **Genre *unique***
   Genre *unique* sebanyak 32 data, diantaranya Action, Adventure, Animation, Aniplex, BROSTA TV, Carousel Productions, Comedy, Crime, Documentary, Drama, Family, Fantasy, Foreign, GoHands, History, Horror, Mardock Scramble Production Committee, Music, Mystery, Odyssey Media, Pulser Productions, Rogue State, Romance, Science Fiction, Sentai Filmworks, TV Movie, Telescene Film Group Productions, The Cartel, Thriller, Vision View Entertainment, War, Western.

3. **Top 10 genre**
   ![Top 10 Genre](https://github.com/user-attachments/assets/8266dcf2-9a1b-4aa1-8557-673081c5281a)

   Gambar di atas menunjukkan genre film yang paling populer. Drama dan Komedi merupakan genre yang paling sering muncul, diikuti oleh Thriller, Romance, dan Action. Genre-genre ini menunjukkan preferensi umum dalam produksi film.

5. **Distribusi tahun rilis**
   ![Distribusi Tahun Rilis](https://github.com/user-attachments/assets/0a20f16d-26f4-4462-9489-c321ef071121)

   Gambar di atas menunjukkan jumlah film yang dirilis setiap tahun dari akhir abad ke-19 hingga awal abad ke-21. Terlihat ada peningkatan jumlah film yang signifikan sejak pertengahan abad ke-20, mencapai puncaknya di tahun 2000-an, lalu sedikit menurun setelahnya.

7. **Top 10 bahasa asli film**
   ![Top 10 Bahasa Asli Film](https://github.com/user-attachments/assets/a0956811-560e-4e84-a99b-9ae5462dc951)

   Gambar di atas menunjukkan bahasa asli yang paling umum digunakan dalam film. Bahasa Inggris (ditampilkan sebagai "en") mendominasi dengan jumlah yang jauh lebih besar dibandingkan bahasa lainnya seperti Prancis, Italia, dan Jepang.
   
9. **Distribusi Rata-Rata Rating Film**
   ![Distribusi Rata-Rata Rating Film](https://github.com/user-attachments/assets/682340e3-55cc-41e4-bad2-2b8471aa79e6)

   Gambar di atas menunjukkan bagaimana distribusi nilai rating rata-rata yang diberikan untuk film. Sebagian besar film mendapatkan rating rata-rata sekitar 5 hingga 7, dengan sedikit yang memiliki nilai mendekati 0 atau sempurna di 10.

10. **Informasi statistik**

| Statistic   | revenue       | runtime       | vote_average | vote_count     |
|-------------|---------------|---------------|--------------|----------------|
| count       | 45460.0000    | 45203.0000    | 45460.0000   | 45460.0000     |
| mean        | 11209348.0000 | 94.1282       | 5.6182       | 109.8973       |
| std         | 64332250.0000 | 38.4078       | 1.9242       | 491.3104       |
| min         | 0.0000        | 0.0000        | 0.0000       | 0.0000         |
| 25%         | 0.0000        | 85.0000       | 5.0000       | 3.0000         |
| 50%         | 0.0000        | 95.0000       | 6.0000       | 10.0000        |
| 75%         | 0.0000        | 107.0000      | 6.8000       | 34.0000        |
| max         | 2787965087.0000 | 1256.0000    | 10.0000      | 14075.0000     |

Tabel dari `movies_metadata.describe()` menunjukkan ringkasan statistik dasar untuk beberapa kolom numerik dalam dataset `movies_metadata`, yaitu `revenue`, `runtime`, `vote_average`, dan `vote_count`.

- **Count**: Jumlah data yang tidak kosong untuk setiap kolom.
  - `revenue`: 45,460
  - `runtime`: 45,203
  - `vote_average`: 45,460
  - `vote_count`: 45,460
- **Mean**: Rata-rata nilai dari setiap kolom.
  - `revenue`: 11,209,350
  - `runtime`: 94.13 menit
  - `vote_average`: 5.62
  - `vote_count`: 109.90
- **Std (Standar Deviasi)**: Penyebaran nilai dalam setiap kolom, menunjukkan variasi data.
  - `revenue`: 64,332,250
  - `runtime`: 38.41
  - `vote_average`: 1.92
  - `vote_count`: 491.31
- **Min**: Nilai minimum dalam setiap kolom.
  - `revenue`, `runtime`, dan `vote_average` memiliki nilai minimum 0.
- **25% (Kuartil Pertama)**: 25% dari data memiliki nilai di bawah ini.
  - `runtime`: 85 menit
  - `vote_average`: 5.0
  - `vote_count`: 3 suara
- **50% (Median)**: Nilai tengah dari data.
  - `runtime`: 95 menit
  - `vote_average`: 6.0
  - `vote_count`: 10 suara
- **75% (Kuartil Ketiga)**: 75% dari data memiliki nilai di bawah ini.
  - `runtime`: 107 menit
  - `vote_average`: 6.8
  - `vote_count`: 34 suara
- **Max**: Nilai maksimum di setiap kolom.
  - `revenue`: 2,787,965,000
  - `runtime`: 1256 menit
  - `vote_average`: 10
  - `vote_count`: 14,075 suara


### ratings
1. **Missing Values**<br>
   Dimulai dengan melihat *missing values* pada berkas `ratings`.<br>

| Column    | Missing Values |
|-----------|----------------|
| userId    | 0              |
| movieId   | 0              |
| rating    | 0              |
| timestamp | 0              |

<br>
Tabel diatas menunjukkan tidak adanya *missing values* pada berkas `ratings`.<br>

2. **Distribusi rating**<br>

   ![Distribusi Rating](https://github.com/user-attachments/assets/ab9a954a-7297-4fb0-bb93-f4251ce4e261)

   Gambar di atas menunjukkan distribusi atau sebaran penilaian yang diberikan pengguna untuk berbagai film. Dari grafik ini, kita bisa melihat bahwa kebanyakan rating berada di angka 4 dan 5, yang menunjukkan banyak pengguna memberikan nilai tinggi pada film yang ditonton.

3. **Rata-rata rating berdasarkan tahun**<br>

   ![Rating Rata-rata Berdasarkan Tahun](https://github.com/user-attachments/assets/913e4910-bfeb-4707-a167-efd2c7d59767)

   Gambar di atas menunjukkan rata-rata penilaian film berdasarkan tahun. Terlihat bahwa rata-rata rating film pada awalnya sangat tinggi, tetapi kemudian mengalami penurunan stabil pada sekitar tahun 2000-an dan tetap relatif konsisten sejak saat itu di sekitar angka 3,5.

4. **Informasi statistik**

| Statistic   | userId       | movieId       | rating     | timestamp       |
|-------------|--------------|---------------|------------|-----------------|
| count       | 26024290.0000 | 26024290.0000 | 26024290.0000 | 26024290.0000 |
| mean        | 135037.1000   | 15849.1100    | 3.5281     | 1.1713e+09     |
| std         | 78176.2000    | 31085.2600    | 1.0654     | 2.0529e+08     |
| min         | 1.0000        | 1.0000        | 0.5000     | 7.8965e+08     |
| 25%         | 67164.0000    | 1073.0000     | 3.0000     | 9.9075e+08     |
| 50%         | 135163.0000   | 2583.0000     | 3.5000     | 1.1517e+09     |
| 75%         | 202693.0000   | 6503.0000     | 4.0000     | 1.3576e+09     |
| max         | 270896.0000   | 176275.0000   | 5.0000     | 1.5018e+09     |

Tabel dari `ratings.describe()` menunjukkan ringkasan statistik dasar untuk beberapa kolom numerik dalam dataset `ratings`, yaitu `userId`, `movieId`, `rating`, dan `timestamp`.

- **Count**: Jumlah data yang tidak kosong untuk setiap kolom, yaitu 26,024,290.
- **Mean**: Rata-rata nilai dari setiap kolom.
  - `userId`: 135,037.1
  - `movieId`: 15,849.11
  - `rating`: 3.53
  - `timestamp`: 1,171,258,000
- **Std (Standar Deviasi)**: Penyebaran nilai dalam setiap kolom, menunjukkan variasi data.
  - `userId`: 78,176.2
  - `movieId`: 31,085.26
  - `rating`: 1.07
  - `timestamp`: 205,288,900
- **Min**: Nilai minimum dalam setiap kolom.
  - `userId`: 1
  - `movieId`: 1
  - `rating`: 0.5
  - `timestamp`: 789,652,000
- **25% (Kuartil Pertama)**: 25% dari data memiliki nilai di bawah ini.
  - `userId`: 67,164
  - `movieId`: 1,073
  - `rating`: 3.0
  - `timestamp`: 990,754,500
- **50% (Median)**: Nilai tengah dari data.
  - `userId`: 135,163
  - `movieId`: 2,583
  - `rating`: 3.5
  - `timestamp`: 1,151,716,000
- **75% (Kuartil Ketiga)**: 75% dari data memiliki nilai di bawah ini.
  - `userId`: 202,693
  - `movieId`: 6,503
  - `rating`: 4.0
  - `timestamp`: 1,357,578,000
- **Max**: Nilai maksimum di setiap kolom.
  - `userId`: 270,896
  - `movieId`: 176,275
  - `rating`: 5.0
  - `timestamp`: 1,501,830,000


### ratings_small
1. **Missing Values**<br>
   Dimulai dengan melihat *missing values* pada berkas `ratings_small`.<br>

| Column    | Missing Values |
|-----------|----------------|
| userId    | 0              |
| movieId   | 0              |
| rating    | 0              |
| timestamp | 0              |

<br>
Tabel diatas menunjukkan tidak adanya *missing values* pada berkas `ratings_small`.<br>

2. **Distribusi rating**<br>

   ![Distribusi Rating](https://github.com/user-attachments/assets/0d37c167-d3fd-4bd7-9fb7-292cc95e7af6)

   Gambar di atas menunjukkan distribusi rating yang diberikan pengguna terhadap film. Kebanyakan film mendapatkan rating antara 3 hingga 4, dengan sedikit film yang mendapat rating sangat rendah (1) atau sangat tinggi (5).
   
3. **Rata-rata rating berdasarkan tahun**
   ![Rating Rata-rata Berdasarkan Tahun](https://github.com/user-attachments/assets/61a95883-a6e7-4ccb-bd12-75be2304e658)

   Gambar di atas menunjukkan perubahan rata-rata rating film dari tahun ke tahun, dengan tren yang cenderung stabil pada nilai sekitar 3,5 sejak tahun 2000.

4. **Informasi statistik**
   
| Statistic   | userId      | movieId       | rating     | timestamp       |
|-------------|-------------|---------------|------------|-----------------|
| count       | 100004.0000 | 100004.0000   | 100004.0000 | 100004.0000    |
| mean        | 347.0113    | 12548.6644    | 3.5436     | 1.1296e+09     |
| std         | 195.1638    | 26369.1990    | 1.0581     | 1.9169e+08     |
| min         | 1.0000      | 1.0000        | 0.5000     | 7.8965e+08     |
| 25%         | 182.0000    | 1028.0000     | 3.0000     | 9.6585e+08     |
| 50%         | 367.0000    | 2406.5000     | 4.0000     | 1.1104e+09     |
| 75%         | 520.0000    | 5418.0000     | 4.0000     | 1.2962e+09     |
| max         | 671.0000    | 163949.0000   | 5.0000     | 1.4766e+09     |


 Tabel dari `ratings_small.describe()` menunjukkan ringkasan statistik dasar untuk beberapa kolom numerik dalam dataset `ratings_small`, yaitu `userId`, `movieId`, `rating`, dan `timestamp`.

- **Count**: Jumlah data yang tidak kosong untuk setiap kolom, yaitu 100,004.
- **Mean**: Rata-rata nilai dari setiap kolom.
  - `userId`: 347.01
  - `movieId`: 12,548.66
  - `rating`: 3.54
  - `timestamp`: 1,129,639,000
- **Std (Standar Deviasi)**: Penyebaran nilai dalam setiap kolom, menunjukkan variasi data.
  - `userId`: 195.16
  - `movieId`: 26,369.20
  - `rating`: 1.06
  - `timestamp`: 191,685,800
- **Min**: Nilai minimum dalam setiap kolom.
  - `userId`: 1
  - `movieId`: 1
  - `rating`: 0.5
  - `timestamp`: 789,652,000
- **25% (Kuartil Pertama)**: 25% dari data memiliki nilai di bawah ini.
  - `userId`: 182
  - `movieId`: 1,028
  - `rating`: 3.0
  - `timestamp`: 965,487,800
- **50% (Median)**: Nilai tengah dari data.
  - `userId`: 367
  - `movieId`: 2,406.5
  - `rating`: 4.0
  - `timestamp`: 1,110,422,000
- **75% (Kuartil Ketiga)**: 75% dari data memiliki nilai di bawah ini.
  - `userId`: 520
  - `movieId`: 5,418
  - `rating`: 4.0
  - `timestamp`: 1,296,192,000
- **Max**: Nilai maksimum di setiap kolom.
  - `userId`: 671
  - `movieId`: 163,949
  - `rating`: 5.0
  - `timestamp`: 1,476,641,000

# Data Preperation

Pada tahap ini dilakukan persiapan data sebelum dilanjutkan ke modeling, yaitu *Data Preprocessing*. Tahap tersebut dilakukan dengan tujuan untuk mempersiapkan data dengan diolah sedemikian rupa untuk dapat dimasukkan ke proses berikutnya. *Data Preprocessing* dilakukan dua bagian untuk dua teknik sistem rekomendasi, untuk CBF dilakukan perubahan tipe data menjadi numerik (`int`), selanjutnya dilakukan ekstraksi pada fitur yang dipilih, setelah di ekstraksi dilakukan penggabungan fitur yang telah diekstraksi menjadi satu fitur. Untuk Collaborative Filtering dilakukan perubahan tipe data dan mengganti nama fiturnya dengan tujuan nantinya akan menggabungkan beberapa fitur dari berkas lain. setelah digabung, dilihat *missing values* jika terdapat maka akan dihapus. Setelah dilakukan pembersihan data pada CBF, berikutnya adalah merubah fitur menjadi representasi numerik dikarenakan dengan representasi numerik ini dapat menggunakan metrik kesamaan seperti cosine similarity untuk menghitung seberapa mirip suatu film dengan film lain berdasarkan `content_features`-nya. Misalnya, dua film dengan genre dan pemeran yang mirip akan memiliki vektor yang relatif serupa, sehingga *similarity* *score*-nya akan tinggi.

## Data Preprocessing
### Content-Based Filtering
1. Proses yang pertama dilakukan untuk CBF ialah merubah tipe data fitur `id` pada berkas `movies_metadata` dan berkas `credits` menjadi `int`.
```python
movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce').astype('Int64')
```
2. Ekstraksi fitur `genre` pada `movies_metadata` dan fitur `cast` dan `crew` pada `credits`. Hal tersebut dilakukan untuk mengambil bagian yang penting pada ketiga fitur tersebut diantaranya mengambil nama genre, nama pemain utama, dan sutradara atau *director*.<br>

| id    | cast              | director         |
|-------|-------------------|------------------|
| 862   | Tom Hanks         | John Lasseter    |
| 8844  | Robin Williams    | Joe Johnston     |
| 15602 | Walter Matthau    | Howard Deutch    |
| 31357 | Whitney Houston   | Forest Whitaker  |
| 11862 | Steve Martin      | Charles Shyer    |

3. Menggabungkan hasil ekstraksi menjadi satu fitur
   Setelah mendapatkan hasil ekstraksi fitur, dilakukan penggabungan ketiga fitur tersebut menjadi satu fitur bernama `content_features`.
   
```python
   # Menggabungkan kolom 'genres', 'cast', dan 'director' menjadi satu fitur bernama content_features
movies_metadata['genres'] = movies_metadata['genres'].apply(lambda x: ' '.join(x))  # Menggabungkan daftar genre menjadi string
movies_metadata['content_features'] = movies_metadata['genres'] + ' ' + credits['cast'] + ' ' + credits['director']
```

Hasilnya sebagai berikut:<br>

| title                   | content_features                                                  |
|-------------------------|-------------------------------------------------------------------|
| Toy Story               | Animation Comedy Family Tom Hanks John Lasseter                   |
| Jumanji                 | Adventure Fantasy Family Robin Williams Joe Johnston              |
| Grumpier Old Men        | Romance Comedy Walter Matthau Howard Deutch                       |
| Waiting to Exhale       | Comedy Drama Romance Whitney Houston Forest Whitaker              |
| Father of the Bride II  | Comedy Steve Martin Charles Shyer                                 |

Tabel diatas menunjukkan setelah menjalankan *script* sebelumnya, tiga fitur (`genre`, `cast`, dan `director`) digabung menjadi satu fitur.

## Collaborative Filtering
1. Merubah tipe data fitur `id` menjadi numerik atau `int`
```python
ratings_data['movieId'] = pd.to_numeric(ratings_data['movieId'], errors='coerce').astype('Int64')
```
2. Merubah nama fitur `id` menjadi `movieId` pada berkas `movies_metadata` untuk mencocokkan penggabungan data ke berkas `ratings_small`.
```python
# Mengganti nama kolom 'id' di movies_metadata menjadi 'movieId' untuk pencocokan
movies_metadata = movies_metadata.rename(columns={'id': 'movieId'})

# Pilih fitur (movieId, original_title, genres) dari movies_metadata
movies_metadata = movies_metadata[['movieId', 'original_title', 'genres']]

# Gabungkan nama film ke dalam ratings_data berdasarkan movieId
ratings_data = ratings_data.merge(movies_metadata, on='movieId', how='left')
```
Hasilnya sebagai berikut:

| userId | movieId | rating | timestamp  | original_title | genres |
|--------|---------|--------|------------|----------------|--------|
| 1      | 31      | 2.5    | 1260759144 | NaN            | NaN    |
| 1      | 1029    | 3.0    | 1260759179 | NaN            | NaN    |
| 1      | 1061    | 3.0    | 1260759182 | NaN            | NaN    |
| 1      | 1129    | 2.0    | 1260759185 | NaN            | NaN    |
| 1      | 1172    | 4.0    | 1260759205 | NaN            | NaN    |

3. Menghapus *missing values*
   Melihat adanya nilai NaN pada fitur `original_title` dan `genres` pada berkas `ratings_small`. Maka dari itu akan dilakukan penghapusan
   
| Column          | Missing Values |
|-----------------|----------------|
| userId          | 0              |
| movieId         | 0              |
| rating          | 0              |
| timestamp       | 0              |
| original_title  | 0              |
| genres          | 0              |<br>

4. Konversi dan Normalisasi Rating
   Proses ini akan mempersiapkan data sebelum dibagi dengan mengonversi `userId` dan `movieId` ke format numerik yang dapat digunakan oleh model, mengonversi `rating` ke tipe `float` dan pemetaan.
```python
# Mengubah `userId` dan `movieId` menjadi list tanpa nilai yang sama
user_ids = ratings_data['userId'].unique().tolist()
movie_ids = ratings_data['movieId'].unique().tolist()

# Melakukan encoding `userId` dan `movieId`
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}

# Mapping angka kembali ke `userId` dan `movieId`
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}

# Pemetaan `userId` dan `movieId` ke dalam kolom `user` dan `movie`
ratings_data['user'] = ratings_data['userId'].map(user_to_user_encoded)
ratings_data['movie'] = ratings_data['movieId'].map(movie_to_movie_encoded)

# Mengubah `rating` menjadi float
ratings_data['rating'] = ratings_data['rating'].astype(np.float32)
```

   

6. Pembagian Data
Pada proses ini dilakukan pembagian data, data akan dibagi menjadi 80:20, 80% digunakan untuk melatih model, sedangkan 20% sisanya digunakan untuk menguji kinerja model pada data yang belum dilihat.

```python
# Split Data menjadi Training dan Validation (80:20)
x = ratings_data[['user', 'movie']].values
y = ratings_data['rating_normalized'].values

train_indices = int(0.8 * ratings_data.shape[0])
x_train, x_val = x[:train_indices], x[train_indices:]
y_train, y_val = y[:train_indices], y[train_indices:]
```
*SCript* diatas mengambil kolom `user` dan `movie` sebagai fitur dalam variabel `x`, serta kolom `rating_normalized` sebagai target dalam variabel `y`.
<br>
Dengan pembagian ini, model dapat dilatih menggunakan data x_train dan y_train, lalu diuji menggunakan x_val dan y_val untuk mengevaluasi kinerjanya pada data yang belum pernah dilihat sebelumnya.     

## TF-IDF
Setelah *preprocessing* **Content-Based Filtering** pada berkas `movies_metadata`, dilakukan representasi numerik menggunakan TF-IDF, dilakukan proses ini dikarenakan fitur `content_features` berisi informasi teks gabungan dari genre, pemeran, dan sutradara film. Algoritma rekomendasi perlu mengetahui seberapa sering kata-kata tertentu muncul atau seberapa relevan kata tersebut di seluruh data untuk mengukur kesamaan antar film.

```python
# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Fit dan transform 'content_features' menjadi matriks fitur TF-IDF
tfidf_matrix = tfidf.fit_transform(movies_metadata['content_features'])
```

# Modeling
Tahapan ini adalah membangun sistem rekomendasi dengan dua teknik, diantaranya Content-Based Filtering dan Collaborative Filtering, tahap ini juga dilakukan pengujian model setelah dibangun guna melihat hasil model.

## Content-Based Filtering
### Membangun model
Memasuki tahap pemodelan untuk CBF. Langkah pertama adalah menghitung kesamaan antar film berdasarkan fitur konten yang telah dihasilkan. Dengan menggunakan cosine_similarity, setiap film akan dibandingkan satu sama lain untuk mendapatkan skor kesamaan berdasarkan fitur kontennya.
```python
# Menghitung kesamaan kosinus antar film
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Melihat bentuk matriks kesamaan untuk memastikan
print("Bentuk matriks kesamaan:", cosine_sim.shape)
```
Matriks kesamaan yang dihasilkan memiliki ukuran (45466, 45466), yang menunjukkan adanya 45.466 film dalam dataset. Setiap nilai dalam matriks ini merepresentasikan tingkat kesamaan antara dua film tertentu. Matriks kesamaan ini akan dimanfaatkan untuk memberikan rekomendasi film yang mirip kepada pengguna.

### Pengujian
Menerapkan fungsi rekomendasi berbasis konten untuk memberikan daftar film yang mirip dengan film yang dipilih berdasarkan fitur kontennya. Dalam fungsi recommend, film yang sesuai dicari dengan judul yang diberikan, kemudian menghitung kesamaan berdasarkan matriks kesamaan yang telah dibuat. Fungsi ini akan menampilkan daftar judul film beserta fitur konten yang mirip, diurutkan berdasarkan tingkat kemiripannya, hasilnya sebagai berikut:

Rekomendasi untuk 'Toy Story 3':<br>

Title: Larry Crowne<br>
Content Features: Comedy Romance Drama Tom Hanks Tom Hanks

Title: Toy Story<br>
Content Features: Animation Comedy Family Tom Hanks John Lasseter

Title: Toy Story 2<br>
Content Features: Animation Comedy Family Tom Hanks John Lasseter

Title: The Legend of Mor'du<br>
Content Features: Animation Family Tom Hanks Steve Purcell

Title: The Polar Express<br>
Content Features: Adventure Animation Family Fantasy Tom Hanks Robert Zemeckis

Hasil rekomendasi untuk film yang dipilih, misalnya "Toy Story 3," menampilkan daftar film yang mirip berdasarkan fitur konten, seperti genre, pemeran utama, dan sutradara. Setiap film yang direkomendasikan menyertakan judul dan detail fitur kontennya, memungkinkan pengguna melihat kesamaan antara film yang direkomendasikan dan film asli yang dipilih.

## Collaborative Filtering
### Membangun model
Proses membangun model rekomendasi berbasis embedding dengan membuat `class` **RecommederNet** dari Keras, di mana embedding layer dibuat untuk user dan movie, dan hasil prediksi diperoleh melalui operasi dot product antara vektor embedding `user` dan `movie`, ditambah bias, dan diaktifkan menggunakan fungsi sigmoid. Kemudian mengkompilasi dan melatih model rekomendasi. Model dikompilasi menggunakan fungsi loss BinaryCrossentropy, optimizer Adam dengan learning rate 0.001, dan metrik evaluasi RootMeanSquaredError. Kemudian, model dilatih pada data training dengan batch size 8 selama 15 epoch dan divalidasi menggunakan data validation.
```python
# Training model
history = model.fit(
    x_train,
    y_train,
    batch_size=8,
    epochs=15,
    validation_data=(x_val, y_val)
```
### Pengujian
Setelah model dibangun, dilakukan pengujian model sistem rekomendasi, hasilnya sebagai berikut

Top 5 Film dengan Rating Tertinggi yang Diberikan oleh User 132:

| No. | Judul Film                        | Movie ID | Rating | Genre                                   |
|-----|-----------------------------------|----------|--------|-----------------------------------------|
| 1   | The Great American Girl Robbery   | 46578    | 5.0    | ['Comedy', 'Crime']                     |
| 2   | Mr. Smith Goes to Washington      | 3083     | 5.0    | ['Comedy', 'Drama']                     |
| 3   | Get Shorty                        | 8012     | 5.0    | ['Comedy', 'Thriller', 'Crime']         |
| 4   | Montag kommen die Fenster         | 4979     | 5.0    | ['Drama']                               |
| 5   | Olga's House of Shame             | 30810    | 5.0    | ['Crime', 'Drama']                      |

Rekomendasi untuk User ID 132:

| No. | Judul Film                        | Movie ID | Genre                                     |
|-----|-----------------------------------|----------|-------------------------------------------|
| 1   | Leiutajateküla Lotte              | 1260     | ['Adventure', 'Animation', 'Comedy', 'Family'] |
| 2   | Lonely Hearts                     | 1252     | ['Drama', 'Thriller', 'Crime', 'Romance'] |
| 3   | Wuthering Heights                 | 98491    | ['Drama', 'Romance']                      |
| 4   | The Million Dollar Hotel          | 318      | ['Drama', 'Thriller']                     |
| 5   | Hannibal Rising                   | 1248     | ['Crime', 'Drama', 'Thriller']            |
| 6   | The Talented Mr. Ripley           | 1213     | ['Thriller', 'Crime', 'Drama']            |
| 7   | 빈집                             | 1280     | ['Drama', 'Romance', 'Crime']             |
| 8   | The Good Shepherd                 | 1247     | ['Drama', 'Thriller', 'History']          |
| 9   | Flags of Our Fathers              | 3683     | ['War', 'Drama', 'History']               |
| 10  | Bean                              | 1281     | ['Comedy']                                |

Tabel diatas merupakan hasil dari model sistem rekomendasi dengan teknik Collaborative Filtering, ditampilkan data user dengan 5 teratas *rating* oleh User ID 132 dan hasil rekomendasi untuk User ID 132.

# Evaluasi
Tahap ini dilakukan evaluasi guna melihat performa model dalam merekomndasikan film untuk pengguna.
## Content-Based Filtering
Metrik yang digunakan untuk mengukur performa model CBF adalah precision (didapat dari forum diskusi), berikut formulanya:

$Precision = \frac{RekomendasiYangRelevan}{TotalItemYangdiRekomendasikan}$<br>

Metrik Precision pada sistem rekomendasi mengukur seberapa akurat rekomendasi yang diberikan dengan hanya mempertimbangkan item yang relevan untuk pengguna. Rumusnya adalah rasio antara jumlah rekomendasi yang relevan terhadap total item yang direkomendasikan. Dengan kata lain, precision menunjukkan seberapa besar proporsi dari item yang direkomendasikan yang benar-benar relevan bagi pengguna.

Mengambil hasil pengujian CBF sebelumnya yaitu:

Rekomendasi untuk 'Toy Story 3':<br>

Title: Larry Crowne<br>
Content Features: Comedy Romance Drama Tom Hanks Tom Hanks

Title: Toy Story<br>
Content Features: Animation Comedy Family Tom Hanks John Lasseter

Title: Toy Story 2<br>
Content Features: Animation Comedy Family Tom Hanks John Lasseter

Title: The Legend of Mor'du<br>
Content Features: Animation Family Tom Hanks Steve Purcell

Title: The Polar Express<br>
Content Features: Adventure Animation Family Fantasy Tom Hanks Robert Zemeckis

Hasil diatas akan diuji genrenya:

Larry Crowne - Comedy Romance Drama (Tidak relevan, genre berbeda)<br>
Toy Story - Animation Comedy Family (Relevan, merupakan film dalam seri yang sama)<br>
Toy Story 2 - Animation Comedy Family (Relevan, merupakan film dalam seri yang sama)<br>
The Legend of Mor'du - Animation Family (Relevan, genre mirip dan memiliki elemen animasi keluarga)<br>
The Polar Express - Adventure Animation Family Fantasy (Relevan, genre mirip dengan elemen animasi dan keluarga)<br>

Pernyataan hasil genre diatas dapat dimasukkan ke formula sebagai berikut:

$Precision = \frac{RekomendasiYangRelevan}{TotalItemYangdiRekomendasikan} = \frac{4}{5} = {0.8}$<br>

Hasil *precision* yang didapat adalah 0.8 atau 80% jika menghitung dari genrenya.

## Collaborative Filtering

Berdasarkan model *machine learning* yang sudah dibangun menggunakan *embedding layer* dengan `Adam` sebagai *optimizer* dan *binary crossentropy loss function*, metrik yang digunakan adalah *Root Mean Squared Error* (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut:

$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$

Di mana, nilai $n$ merupakan jumlah *dataset*, nilai $y_i$ adalah nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya terdahap $i$ sebagai urutan data dalam *dataset*.

Metrik di atas cara bekerjanya seperti dibawah ini:
Formula **Root Mean Squared Error (RMSE)** mengukur seberapa dekat prediksi model dengan nilai sebenarnya. Langkah-langkah kerjanya adalah sebagai berikut:

1. **Menghitung Selisih Setiap Pasangan Nilai**  
   Untuk setiap pasangan nilai aktual $y_i$ dan nilai prediksi $y\\_pred_i$, hitung selisihnya $y_i - y\\_pred_i$. Ini menunjukkan seberapa jauh prediksi dari nilai aktual.

2. **Mengkuadratkan Selisih**  
   Kuadratkan setiap selisih $(y_i - y\\_pred_i)^2$ untuk memastikan semua nilai positif dan memberikan penalti lebih besar pada kesalahan yang lebih besar.

3. **Menjumlahkan Semua Selisih Kuadrat**  
   Jumlahkan semua nilai selisih kuadrat dari setiap pasangan nilai dalam dataset.

4. **Menghitung Nilai Rata-Rata**  
   Bagi jumlah total selisih kuadrat dengan jumlah data $n$ untuk mendapatkan rata-rata kesalahan kuadrat.

5. **Mengambil Akar Kuadrat dari Rata-Rata**  
   Terakhir, ambil akar kuadrat dari rata-rata kesalahan kuadrat untuk mendapatkan nilai RMSE.

Semakin kecil nilai RMSE, semakin baik kinerja model, karena prediksi mendekati nilai observasi atau nilai sebenarnya.

![RMSE](https://github.com/user-attachments/assets/9d82cb5b-33e6-43e4-a80b-7209e0eb78c0)

Visualisasi di atas menunjukkan penurunan Root Mean Squared Error (RMSE) pada data pelatihan dan data validasi selama 15 epoch. Grafik memperlihatkan bahwa error pada data pelatihan terus menurun, sementara error pada data validasi awalnya menurun namun kemudian stabil, yang menunjukkan bahwa model belajar dari data dengan cukup baik tanpa overfitting.


# Referensi<br>
[^1]: [M. A. Maulana, Y. Wihardi, and E. Piantari, "Mesin Rekomendasi Film Menggunakan Metode Deep Autoencoder," in *Prodi Studi Ilmu Komputer Departemen Pendidikan Ilmu Komputer Fakultas Pendidikan Matematika dan Ilmu Pengetahuan Alam Universitas Pendidikan Indonesia*, Bandung, Indonesia, 2024.
]()
[^2]: [IMDb. "IMDb Statistics." Accessed: Nov. 5, 2024](https://www.imdb.com/pressroom/stats/).
[^3]: [C. A. Gomez-Uribe and N. Hunt, "The Netflix Recommender System: Algorithms, Business Value, and Innovation," *ACM Transactions on Management Information Systems (TMIS)*, vol. 6, no. 4, pp. 1–19, 2016.](https://dl.acm.org/doi/pdf/10.1145/2843948)

