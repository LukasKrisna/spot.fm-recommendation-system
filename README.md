
# Laporan Proyek Machine Learning - Sistem Rekomendasi Musik

#### Disusun oleh: Lukas Krisna

## Project Overview

Industri musik digital telah merevolusi cara kita menikmati musik, dengan platform streaming seperti [Spotify](https://open.spotify.com/) dan Last.fm yang menyediakan akses ke katalog lagu yang sangat besar. Dalam lautan musik ini, pengguna seringkali kesulitan menemukan lagu baru yang sesuai dengan selera mereka. Sistem rekomendasi memainkan peran krusial dalam mengatasi masalah ini dengan menyarankan item (dalam hal ini, lagu) yang mungkin disukai pengguna. Sistem ini tidak hanya meningkatkan pengalaman pengguna dengan personalisasi tetapi juga membantu artis menjangkau audiens yang lebih luas dan platform untuk meningkatkan keterlibatan pengguna.

Pentingnya sistem rekomendasi musik yang efektif telah diakui secara luas. Penelitian menunjukkan bahwa rekomendasi yang dipersonalisasi dapat secara signifikan meningkatkan kepuasan dan retensi pengguna. Sebagai contoh, [Gao et al. (2022)](https://dl.acm.org/doi/pdf/10.1145/3568022) membahas pentingnya dan tantangan dalam sistem rekomendasi musik, menyoroti bagaimana berbagai pendekatan dapat dimanfaatkan untuk memberikan saran yang relevan. Lebih lanjut, pengembangan model seperti yang diusulkan oleh [He et al. (2017)](https://arxiv.org/pdf/1708.05031) untuk rekomendasi dari data implisit (seperti jumlah putar) menunjukkan evolusi berkelanjutan dalam bidang ini. Proyek ini bertujuan untuk mengembangkan dan mengevaluasi dua pendekatan populer dalam sistem rekomendasi musik: _Content-Based Filtering_ dan _Collaborative Filtering_, menggunakan dataset yang mencerminkan interaksi pengguna dengan lagu.

## Business Understanding

Nilai bisnis inti dari proyek sistem rekomendasi musik ini terletak pada kemampuannya untuk meningkatkan pengalaman pengguna pada platform musik. Peningkatan pengalaman ini, pada gilirannya, dapat mengarah pada peningkatan keterlibatan pengguna, retensi pelanggan yang lebih tinggi, dan potensi peningkatan pendapatan melalui langganan atau iklan yang ditargetkan. Dengan menyediakan rekomendasi lagu yang akurat dan dipersonalisasi, platform dapat menjadi lebih menarik dan berharga bagi penggunanya.

### Problem Statements

Berdasarkan latar belakang tersebut, rincian permasalahan yang akan dibahas pada proyek ini adalah:

1.  Bagaimana cara membangun sistem yang dapat merekomendasikan lagu kepada pengguna berdasarkan kemiripan konten lagu (genre) dengan lagu yang pernah mereka sukai?
2.  Bagaimana cara membangun sistem yang dapat merekomendasikan lagu kepada pengguna berdasarkan pola pendengaran pengguna lain yang memiliki selera serupa?
3.  Metode manakah (_Content-Based Filtering_ vs. _Collaborative Filtering_) yang lebih efektif atau memberikan jenis rekomendasi yang berbeda dalam konteks dataset yang digunakan?

### Goals

Berdasarkan _problem statements_, berikut adalah tujuan yang ingin dicapai dalam proyek ini:

1.  Mengembangkan model sistem rekomendasi menggunakan _Content-Based Filtering_ yang mampu menyarankan lagu berdasarkan atribut musik seperti genre.
2.  Mengembangkan model sistem rekomendasi menggunakan _Collaborative Filtering_ yang mampu menyarankan lagu berdasarkan riwayat interaksi pengguna (jumlah putar).
3.  Mengevaluasi dan membandingkan kinerja kedua model rekomendasi tersebut menggunakan metrik yang sesuai.

### Solution Approach

Untuk mencapai tujuan-tujuan di atas, solusi yang diajukan adalah sebagai berikut:

1.  **Content-Based Filtering**:
    * Melakukan pra-pemrosesan pada data informasi musik, khususnya pada fitur genre.
    * Menggunakan TF-IDF Vectorizer untuk mengubah data tekstual genre menjadi representasi numerik.
    * Menghitung kemiripan antar lagu menggunakan _cosine similarity_ berdasarkan fitur genre.
    * Menghasilkan top-N rekomendasi lagu berdasarkan lagu input.
2.  **Collaborative Filtering**:
    * Melakukan pra-pemrosesan pada data riwayat pendengaran pengguna, termasuk normalisasi _playcount_.
    * Melakukan encoding pada ID pengguna dan ID lagu.
    * Mengembangkan model _neural network_ dengan _embedding layers_ untuk pengguna dan lagu.
    * Melatih model untuk memprediksi preferensi pengguna terhadap lagu berdasarkan interaksi sebelumnya.
    * Menghasilkan top-N rekomendasi lagu untuk pengguna tertentu.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Million Song Dataset an Echonest subset with additional data from last.fm and spotify" yang diperoleh dari Kaggle. Dataset ini berisi informasi tentang lagu dan riwayat pendengaran pengguna.

-   **Sumber Data:** [https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm/data](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm/data)
    Dalam proyek ini, dua file utama dari dataset tersebut digunakan:
    1.  `Music Info.csv`: Berisi informasi detail tentang setiap lagu.
    2.  `User Listening History.csv`: Berisi riwayat interaksi pengguna dengan lagu, termasuk jumlah putar (_playcount_).

-   **Informasi Dataset:**
    -   `User Listening History.csv`: 9.711.301 baris, namun disampling menjadi 100.000 baris.
    -   `Music Info.csv`: 50.683 baris informasi lagu.
    -   Setelah sampling dan pemfilteran awal pada `user_history_df_reduced`:
        -   Jumlah pengguna unik: 991
        -   Jumlah lagu unik yang didengarkan: 59.944
        -   Statistik _playcount_ sebelum filter:
            -   mean: 2.63
            -   std: 5.7
            -   min: 1.0
            -   25%: 1.0
            -   50%: 1.0
            -   75%: 2.0
            -   max: 2948.0
        -   Dataset kemudian difilter untuk _playcount_ <= 6.

### Deskripsi Variabel

Berikut adalah deskripsi variabel yang relevan dari `Music Info.csv` dan `User Listening History.csv` yang digunakan dalam proyek:

**Dari `Music Info.csv`:**

| Variabel     | Keterangan                                                                 | Status Penggunaan dalam Proyek |
| :----------- | :------------------------------------------------------------------------- | :----------------------------- |
| `track_id`   | ID unik untuk setiap lagu.                                                 | Digunakan (Kunci utama)        |
| `name`       | Nama atau judul lagu.                                                      | Digunakan                      |
| `artist`     | Nama artis yang membawakan lagu.                                           | Digunakan                      |
| `spotify_preview_url`| URL pratinjau lagu di Spotify. | Dihapus | 
| `spotify_id` | ID unik lagu di Spotify. | Dihapus |
| `tags`       | Tag atau beberapa genre yang diasosiasikan dengan lagu, dipisahkan oleh koma.       | Digunakan (diproses menjadi genre tunggal) |
| `genre` | Genre tunnggal lagu. | Dihapus |
| `year` | Tahun rilis lagu. | Dihapus | 
| `duration_ms` | Durasi lagu dalam milidetik. | Dihapus |
| `danceability`     | Menggambarkan seberapa cocok sebuah lagu untuk menari berdasarkan kombinasi elemen musik termasuk tempo, stabilitas ritme, kekuatan ketukan, dan keteraturan keseluruhan. Nilai 0.0 paling tidak bisa ditarikan dan 1.0 paling bisa ditarikan.                                            | Dihapus                      |
| `energy`           | Merupakan ukuran persepsi dari intensitas dan aktivitas. Biasanya, trek yang energik terasa cepat, keras, dan berisik. Nilai dari 0.0 hingga 1.0.                                                                                                                                         | Dihapus                      |
| `key`              | Kunci nada keseluruhan dari trek. Direpresentasikan dalam notasi Pitch Class standar (misalnya 0 = C, 1 = C♯/D♭, 2 = D, dst.).                                                                                                                                                            | Dihapus                      |
| `loudness`         | Kekerasan keseluruhan sebuah trek dalam desibel (dB). Nilai biasanya berkisar antara -60 dan 0 db.                                                                                                                                                                                        | Dihapus                      |
| `mode`             | Mode (mayor atau minor) dari sebuah trek. Mayor direpresentasikan oleh 1 dan minor oleh 0.                                                                                                                                                                                                | Dihapus                      |
| `speechiness`      | Mendeteksi keberadaan kata-kata yang diucapkan dalam sebuah trek. Semakin eksklusif rekaman itu mirip ucapan (misalnya acara bincang-bincang, buku audio, puisi), semakin mendekati 1.0 nilai atributnya.                                                                                 | Dihapus                      |
| `acousticness`     | Ukuran kepercayaan dari 0.0 hingga 1.0 apakah trek tersebut akustik.                                                                                                                                                                                                                      | Dihapus                      |
| `instrumentalness` | Memprediksi apakah sebuah trek tidak mengandung vokal. Suara "Ooh" dan "aah" dianggap instrumental dalam konteks ini. Semakin mendekati 1.0 nilai instrumentalness, semakin besar kemungkinan trek tersebut tidak mengandung konten vokal.                                                | Dihapus                      |
| `liveness`         | Mendeteksi keberadaan audiens dalam rekaman. Nilai liveness yang lebih tinggi menunjukkan kemungkinan yang meningkat bahwa trek tersebut dibawakan secara langsung.                                                                                                                       | Dihapus                      |
| `valence`          | Ukuran dari 0.0 hingga 1.0 yang menggambarkan positivitas musik yang disampaikan oleh sebuah trek. Trek dengan valensi tinggi terdengar lebih positif (misalnya bahagia, ceria, euforia), sedangkan trek dengan valensi rendah terdengar lebih negatif (misalnya sedih, tertekan, marah). | Dihapus                      |
| `tempo`            | Perkiraan tempo keseluruhan sebuah trek dalam ketukan per menit (BPM).  | Dihapus                        |
| `time_signature`   | Perkiraan birama keseluruhan suatu lintasan. Birama adalah konvensi notasi untuk menentukan berapa banyak ketukan dalam setiap bar (atau ukuran).                                                                                                                                         | Dihapus                        |


**Dari `User Listening History.csv`:**

| Variabel    | Keterangan                                                                                                   | Status Penggunaan dalam Proyek |
| :---------- | :----------------------------------------------------------------------------------------------------------- | :----------------------------- |
| `user_id`   | ID unik untuk setiap pengguna.                                                                               | Digunakan                      |
| `track_id`  | ID unik lagu yang didengarkan oleh pengguna.                                                                 | Digunakan (Kunci penghubung)   |
| `playcount` | Jumlah berapa kali seorang pengguna telah memutar lagu tertentu.                                            | Digunakan (Sebagai rating implisit) |

### Exploratory Data Analysis (EDA) untuk Memahami Karakteristik Data

EDA dilakukan untuk memahami lebih dalam karakteristik dataset.

1.  **Distribusi Playcount:**
    Histogram _playcount_ (setelah difilter <= 6) menunjukkan bahwa sebagian besar lagu diputar dalam jumlah kecil oleh pengguna, yang umum dalam data implisit.
    ```python
    plt.figure(figsize=(10, 6))
    user_history_df_reduced['playcount'].plot(kind='hist', bins=5, title='Distribution of Playcounts') #
    plt.xlabel('Playcount') #
    plt.ylabel('Frequency') #
    plt.grid(axis='y', alpha=0.75) #
    plt.show() #
    ```
    ![distribusi playcount](https://github.com/user-attachments/assets/d9402922-e80e-4172-8746-13adf9269713)
    
    *Insight*: Distribusi _playcount_ yang condong ke kanan menunjukkan bahwa sebagian besar interaksi adalah pemutaran tunggal atau beberapa kali putar. Pemfilteran `playcount <= 6` membantu mengurangi bias dari pengguna yang sangat aktif atau _power users_ pada sejumlah kecil lagu.

2.  **Distribusi Genre:**
    Visualisasi 15 genre teratas memberikan gambaran tentang genre musik yang paling umum dalam dataset.
    ```python
    plt.figure(figsize=(12, 7)) #
    music_info_df['tags_cleaned'].value_counts().head(15).plot(kind='bar', title='Top 15 Genres') #
    plt.xlabel('Genre') #
    plt.ylabel('Number of Tracks') #
    plt.xticks(rotation=45, ha='right') #
    plt.tight_layout() #
    plt.show() #
    ```
    ![distribusi genre](https://github.com/user-attachments/assets/8956d11d-80f5-44be-bdb1-944c9cf1c06a)
    *Insight*: Genre seperti "rock", "electronic", "pop", "metal", dan "indie" tampak dominan. Ini penting untuk _Content-Based Filtering_ karena variasi dan distribusi genre akan mempengaruhi kualitas rekomendasi berbasis genre.

3.  **Distribusi Artis:**
    Visualisasi 15 artis teratas menunjukkan artis dengan jumlah lagu terbanyak dalam dataset.
    ```python
    plt.figure(figsize=(12, 7)) #
    music_info_df['artist'].value_counts().head(15).plot(kind='bar', title='Top 15 Artists') #
    plt.xlabel('Artist') #
    plt.ylabel('Number of Tracks') #
    plt.xticks(rotation=45, ha='right') #
    plt.tight_layout() #
    plt.show() #
    ```
    ![distribusi artis](https://github.com/user-attachments/assets/0afdf14e-a982-42be-8af2-242abcdf9f3d)
    *Insight*: Beberapa artis memiliki kontribusi lagu yang signifikan dalam dataset. Ini dapat mempengaruhi rekomendasi, terutama jika pengguna menunjukkan preferensi untuk artis tertentu.

4.  **Pemeriksaan Nilai Hilang dan Tipe Data:**
    -   Dilakukan pemeriksaan `music_info_df.info()` pada `music_info_df`.

    <br>
    
	<img width="490" alt="Image" src="https://github.com/user-attachments/assets/66ec0a6a-3d77-43a2-b91e-bcd06d09e4ca" />

    <br>

## Data Preparation

Tahapan ini mencakup semua langkah transformasi data yang dilakukan untuk menyiapkan dataset agar sesuai untuk _Content-Based Filtering_ dan _Collaborative Filtering_.

1.  **Pengambilan Sampel Data Riwayat Pengguna:**
    -   `user_history_df` disampling menjadi 100.000 baris (`user_history_df_reduced`).
    -   **Alasan:** Mengurangi ukuran dataset untuk komputasi yang lebih efisien selama pengembangan dan eksperimen, sambil tetap berusaha mempertahankan representasi data yang wajar.
    ```python
    user_history_df_reduced = user_history_df.sample(n=100000, random_state=42)
    ```

2.  **Filter Playcount:**
    -   Interaksi dengan `playcount > 6` dihapus dari `user_history_df_reduced`.
    -   **Alasan:** Mengurangi pengaruh _outlier_ atau _power users_ yang mungkin mendistorsi model _Collaborative Filtering_ dan untuk memfokuskan pada preferensi yang lebih umum.
    ```python
    user_history_df_reduced = user_history_df_reduced[user_history_df_reduced['playcount'] <= 6] 
    ```

3.  **Pembersihan dan Ekstraksi Genre:**
    -   Kolom `tags` pada `music_info_df` diproses untuk mengekstrak tag pertama sebagai genre (`tags_cleaned`).
    -   Genre dibersihkan lebih lanjut dengan menghapus karakter non-alfabetik dan dikonversi menjadi huruf kecil.
    -   Baris dengan `tags_cleaned` yang NaN dihapus.
    -   **Alasan:** Mendapatkan representasi genre yang bersih dan konsisten untuk _Content-Based Filtering_. Menggunakan tag pertama adalah heuristik untuk mendapatkan genre utama.
    ```python
    music_info_df['tags_cleaned'] = music_info_df['tags'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) and x.strip() else np.nan) 
    music_info_df['tags_cleaned'] = music_info_df['tags_cleaned'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)).strip().lower() if pd.notna(x) else np.nan) 
    music_info_df.dropna(subset=['tags_cleaned'], inplace=True) 
    ```

4.  **Pembuatan DataFrame Bersih untuk Informasi Musik (`clean_music_df`):**
    -   Memilih kolom `track_id`, `name`, `artist`, `tags_cleaned` (diganti nama menjadi `genre`).
    -   Menghapus baris dengan nilai NaN pada `name`, `artist`, `genre`.
    -   Menghapus duplikat `track_id`.
    -   **Alasan:** Membuat dataset musik yang ringkas dan bersih khusus untuk digunakan dalam sistem rekomendasi, terutama untuk mencocokkan ID lagu dan mengambil metadata.
    ```python
    clean_music_df = music_info_df[['track_id', 'name', 'artist', 'tags_cleaned']].copy()
    clean_music_df.rename(columns={'tags_cleaned': 'genre'}, inplace=True) 
    clean_music_df.dropna(subset=['name', 'artist', 'genre'], inplace=True) 
    clean_music_df = clean_music_df.drop_duplicates('track_id') 
    ```
    DataFrame ini menjadi dasar untuk _Content-Based Filtering_ dan untuk memperkaya output dari _Collaborative Filtering_.

5. **Pembuatan DataFrame (`music_for_content_based`)**
    - Untuk lebih memastikan struktur data yang akan digunakan oleh model Content-Based Filtering, kolom-kolom dari `clean_music_df` (`track_id`, `name`, `artist`, `genre`) diekstrak menjadi list terpisah.
    - List-list ini kemudian digunakan untuk membuat DataFrame baru, yaitu `music_for_content_based`.
    ```python
    track_ids_cb = clean_music_df['track_id'].tolist()
    track_names_cb = clean_music_df['name'].tolist()
    track_artist_cb = clean_music_df['artist'].tolist()
    track_genres_cb = clean_music_df['genre'].tolist()

    music_for_content_based = pd.DataFrame({
    'track_id': track_ids_cb,
    'track_name': track_names_cb,
    'artist': track_artist_cb,
    'genre': track_genres_cb
    })
    ```

6. **TF-IDF Vectorization pada Genre (untuk Content-Based Filtering):**
    -   Kolom `genre` dari `music_for_content_based` (hasil dari `clean_music_df`) digunakan.
    -   `TfidfVectorizer` mengubah daftar genre menjadi matriks representasi numerik TF-IDF. Setiap baris mewakili sebuah lagu, dan setiap kolom mewakili sebuah istilah genre unik. Nilai dalam matriks menunjukkan seberapa penting sebuah genre untuk sebuah lagu.
    -   **Alasan:** Mengubah fitur tekstual (genre) menjadi format numerik yang dapat digunakan oleh algoritma _machine learning_ untuk menghitung kemiripan.
    ```python
    data_cb = music_for_content_based

    tfidf_vectorizer = TfidfVectorizer() 
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_cb['genre']) 
    ```

6.  **Penggabungan Data untuk Collaborative Filtering (`df_cf`):**
    -   `user_history_df_reduced` digabungkan dengan `music_for_content_based` berdasarkan `track_id`.
    -   **Alasan:** Menggabungkan riwayat interaksi pengguna dengan metadata lagu untuk analisis lebih lanjut dan persiapan data _Collaborative Filtering_.
    ```python
    df_cf = pd.merge(user_history_df_reduced, music_for_content_based[['track_id', 'track_name', 'artist', 'genre']], on='track_id', how='inner') 
    ```

7.  **Encoding User dan Track ID untuk Collaborative Filtering:**
    -   ID pengguna (`user_id`) dan ID lagu (`track_id`) dipetakan ke integer unik.
    -   **Alasan:** Model _machine learning_, terutama _embedding layers_ pada _neural network_, memerlukan input numerik kategorikal.
    ```python
    user_ids = df_cf['user_id'].unique().tolist() 
    user_to_user_encoded = {x: i for i, x in enumerate(user_ids)} 
    user_encoded_to_user = {i: x for i, x in enumerate(user_ids)} 

    track_ids_cf = df_cf['track_id'].unique().tolist() 
    track_to_track_encoded = {x: i for i, x in enumerate(track_ids_cf)} 
    track_encoded_to_track = {i: x for i, x in enumerate(track_ids_cf)} 

    df_cf['user'] = df_cf['user_id'].map(user_to_user_encoded) 
    df_cf['track'] = df_cf['track_id'].map(track_to_track_encoded) 
    ```

8.  **Normalisasi Playcount untuk Collaborative Filtering:**
    -   Nilai `playcount` dikonversi ke float32 dan dinormalisasi ke rentang [0, 1] menggunakan Min-Max scaling.
    -   **Alasan:** Menstabilkan proses pelatihan model _neural network_ dan memastikan bahwa _playcount_ diperlakukan sebagai skor preferensi dalam rentang yang konsisten untuk fungsi aktivasi sigmoid.
    ```python
    df_cf['playcount'] = df_cf['playcount'].values.astype(np.float32) 
    min_playcount = min(df_cf['playcount']) 
    max_playcount = max(df_cf['playcount']) 
    y = df_cf['playcount'].apply(lambda x: (x - min_playcount) / (max_playcount - min_playcount)).values 
    ```

9. **Mengacak Urutan Data untuk Collaborative Filtering:**
	-  Dataset `df_cf` (yang berisi data untuk _Collaborative Filtering_) diacak urutannya.
    -   **Alasan:** Memastikan bahwa saat data dibagi menjadi set pelatihan dan validasi, kedua set tersebut memiliki distribusi data yang representatif dan tidak ada bias urutan yang tidak disengaja yang dapat mempengaruhi pelatihan model.
    ```python
    df_cf = df_cf.sample(frac=1, random_state=42)
    ```
10.  **Pemisahan Data Latih dan Validasi untuk Collaborative Filtering:**
    -   Dataset `df_cf` diacak dan dibagi menjadi data latih (70%) dan data validasi (30%).
    -   **Alasan:** Untuk melatih model _Collaborative Filtering_ dan mengevaluasi kinerjanya pada data yang tidak terlihat selama pelatihan.

        ```python
        df_cf = df_cf.sample(frac=1, random_state=42) 
        x = df_cf[['user', 'track']].values 
        train_indices = int(0.7 * df_cf.shape[0]) 
        x_train, x_val, y_train, y_val = (
            x[:train_indices], 
            x[train_indices:], 
            y[:train_indices], 
            y[train_indices:] 
        )   
        ```

## Modeling and Result

Menggunakan dua pendekatan sistem rekomendasi: _Content-Based Filtering_ dan _Collaborative Filtering_.

### 1. Content-Based Filtering

Pendekatan ini merekomendasikan lagu berdasarkan kemiripan atribut kontennya (dalam hal ini, genre) dengan lagu yang disukai pengguna atau lagu input.

-   **Tahapan:**
    1.  **Perhitungan Cosine Similarity:**
        -   _Cosine similarity_ dihitung antar semua pasangan lagu berdasarkan vektor TF-IDF genre mereka. Hasilnya adalah matriks kemiripan (cosine_sim_df) di mana setiap elemen (i, j) menunjukkan kemiripan antara lagu i dan lagu j.
        ```python
        cosine_sim = cosine_similarity(tfidf_matrix) 
        cosine_sim_df = pd.DataFrame(cosine_sim, index=data_cb['track_name'], columns=data_cb['track_name']) 
        ```
    2.  **Pembuatan Fungsi Rekomendasi (`music_recommendations`):**
        -   Fungsi ini mengambil nama lagu input dan `k` (jumlah rekomendasi) sebagai argumen.
        -   Mencari lagu input dalam matriks kemiripan.
        -   Mengambil `k` lagu teratas yang paling mirip (tidak termasuk lagu input itu sendiri).
        -   Mengembalikan DataFrame berisi nama lagu, artis, dan genre dari lagu-lagu yang direkomendasikan.

-   **Hasil Rekomendasi:**
    <br>
    <br>
    <img width="508" alt="content based recommenndation" src="https://github.com/user-attachments/assets/ad9de01c-2125-40b3-bfd2-2d4885fde68d" />

    Hasil ini menunjukkan bahwa sistem rekomendasi content-based filtering berfungsi dengan baik, di mana untuk lagu "The Revelation" sistem merekomendasikan 5 lagu metal (Crowbar, Megadeth, Rammstein, Metallica), dan untuk lagu "Wonderwall" sistem merekomendasikan 5 lagu rock (9mm Parabellum Bullet, dEUS, Stereophonics, Nine Inch Nails, Swans).

-   **Kelebihan:**
    -   Dapat merekomendasikan item baru yang belum memiliki interaksi pengguna (_cold start_ untuk item).
    -   Rekomendasi bersifat transparan karena didasarkan pada fitur item yang dapat dijelaskan.
    -   Tidak memerlukan data dari pengguna lain.
-   **Kekurangan:**
    -   Kualitas rekomendasi sangat bergantung pada kualitas dan kelengkapan fitur item. Jika fitur genre kurang deskriptif atau terlalu umum, rekomendasi bisa menjadi kurang relevan.
    -   Cenderung merekomendasikan item yang sangat mirip dengan apa yang sudah diketahui pengguna (_serendipity_ rendah).
    -   Membutuhkan _domain knowledge_ untuk _feature engineering_ yang baik.

### 2. Collaborative Filtering

Pendekatan ini merekomendasikan lagu berdasarkan pola perilaku pengguna lain. Asumsinya adalah jika pengguna A memiliki preferensi yang sama dengan pengguna B pada beberapa lagu, maka pengguna A kemungkinan akan menyukai lagu lain yang disukai pengguna B.

-   **Tahapan (Model Neural Network - `RecommenderNet`):**
    1.  **Arsitektur Model:**
        -   Model `RecommenderNet` menggunakan _embedding layers_ untuk pengguna dan lagu. _Embedding_ ini mempelajari representasi vektor laten (fitur tersembunyi) untuk setiap pengguna dan lagu.
        -   Ukuran _embedding_ (`embedding_size`) diatur ke 5.
        -   Bias pengguna dan bias lagu juga ditambahkan sebagai _embedding_.
        -   _Dropout layers_ ditambahkan setelah _embedding_ pengguna dan lagu untuk mencegah _overfitting_.
        -   Interaksi antara pengguna dan lagu dimodelkan menggunakan _dot product_ dari vektor _embedding_ mereka, ditambah dengan bias.
        -   Fungsi aktivasi sigmoid digunakan pada output untuk menghasilkan skor prediksi antara 0 dan 1 (mewakili probabilitas preferensi setelah normalisasi _playcount_).
        ```python
        class RecommenderNet(tf.keras.Model): #
            def __init__(self, num_users, num_track, embedding_size, dropout_rate=0.2, **kwargs): #
                super(RecommenderNet, self).__init__(**kwargs) #
                # ... (definisi layer seperti dalam notebook) ...
            def call(self, inputs): #
                # ... (operasi forward pass seperti dalam notebook) ...
                return tf.nn.sigmoid(x) #
        ```
    2.  **Kompilasi dan Pelatihan Model:**
        -   Model dikompilasi dengan loss `BinaryCrossentropy` (karena output sigmoid dan target ternormalisasi [0,1] dapat diinterpretasikan sebagai probabilitas), optimizer `Adam`, dan metrik `RootMeanSquaredError`.
        -   _Callback_ `EarlyStopping` digunakan untuk menghentikan pelatihan jika `val_root_mean_squared_error` tidak membaik selama beberapa epoch, dan untuk mengembalikan bobot terbaik.
        -   Model dilatih menggunakan `x_train` (pasangan user-track encoded) dan `y_train` (playcount ternormalisasi).
        ```python
        model = RecommenderNet(num_users, num_track, embedding_size=5, dropout_rate=0.2) 
        model.compile( 
            loss = tf.keras.losses.BinaryCrossentropy(), 
            optimizer = keras.optimizers.Adam(learning_rate=0.0005), 
            metrics=[tf.keras.metrics.RootMeanSquaredError()] 
        )
        history = model.fit( 
            x = x_train, 
            y = y_train, 
            batch_size = 32, 
            epochs = 50, 
            validation_data = (x_val, y_val), 
            callbacks=[early_stopping_callback] 
        )
        ```
    3.  **Pembuatan Fungsi Rekomendasi:**
        -   Untuk pengguna tertentu, prediksi skor preferensi dibuat untuk semua lagu yang belum pernah didengarkan oleh pengguna tersebut.
        -   Lagu-lagu diurutkan berdasarkan skor prediksi tertinggi.
        -   Top-N lagu direkomendasikan.

-   **Hasil Rekomendasi:**
    Menampilkan rekomendasi untuk pengguna acak:
    <br>
    <br>
    <img width="660" alt="collaborative result" src="https://github.com/user-attachments/assets/368965a4-1d65-4a3c-8f2e-6873935ec7ce" />
    <br>
    (Output sebenarnya dari kode akan bervariasi tergantung pada pengguna yang disampel).

-   **Kelebihan:**
    -   Tidak memerlukan fitur item secara eksplisit; dapat menemukan pola preferensi yang kompleks dan tak terduga (_serendipity_ lebih tinggi).
    -   Dapat bekerja dengan baik bahkan jika konten item sulit dianalisis.
    -   Seiring bertambahnya data pengguna dan interaksi, kualitas rekomendasi dapat meningkat.
-   **Kekurangan:**
    -   Mengalami masalah _cold start_ untuk pengguna baru dan item baru (item/pengguna tanpa interaksi tidak dapat direkomendasikan atau menerima rekomendasi dengan baik).
    -   Membutuhkan banyak data interaksi pengguna untuk performa yang baik.
    -   Rekomendasi bisa jadi kurang dapat dijelaskan ("black box").
    -   _Sparsity_ data (banyak pengguna hanya berinteraksi dengan sedikit item) dapat menjadi tantangan.

## Evaluation

Metrik evaluasi digunakan untuk mengukur kinerja dari kedua sistem rekomendasi yang telah dikembangkan.

### 1. Metrik Evaluasi untuk Content-Based Filtering

-   **Coverage (Cakupan Katalog):**
    -   **Penjelasan:** _Coverage_ mengukur seberapa banyak dari total item unik dalam katalog yang dapat direkomendasikan oleh sistem. Dalam konteks ini, ini adalah persentase lagu unik dalam dataset yang muncul dalam daftar rekomendasi (misalnya, top-10) ketika rekomendasi dihasilkan untuk sampel lagu dari dataset.
    -   **Rumus:**
        $$\text{Coverage} = \frac{|\text{Unique Items Recommended}|}{|\text{Total Unique Items in Catalog}|} \times 100\%$$
        Di mana:
        -   `Unique Items Recommended` adalah jumlah item unik yang muncul dalam set rekomendasi yang dihasilkan untuk sampel input.
        -   `Total Unique Items in Catalog` adalah jumlah total item unik yang tersedia dalam dataset.
    -   **Cara Kerja:** Sejumlah lagu sampel (`sampled_tracks_for_coverage`) diambil dari dataset. Untuk setiap lagu sampel, top-K rekomendasi dihasilkan. Semua lagu unik yang muncul di rekomendasi ini dikumpulkan. Jumlah lagu unik ini kemudian dibagi dengan total lagu unik dalam dataset keseluruhan (`data_cb`).
    -   **Hasil:**
        ```
        Total unique tracks in dataset: 49556 
        Number of unique tracks recommended (from sample): 770 
        Content-Based System Coverage (Estimated): 1.55%
        ```
        Hasil cakupan sebesar 1.55% ini menunjukkan bahwa model Content-Based Filtering saat ini, yang berfokus pada kemiripan genre, cenderung merekomendasikan sebagian kecil dari total 49.556 lagu unik dalam katalog. Meskipun rekomendasi yang dihasilkan mungkin relevan secara genre, angka ini juga mengindikasikan adanya ruang untuk peningkatan dalam hal eksplorasi katalog dan penemuan lagu yang lebih beragam, mungkin dengan memperkaya fitur konten yang digunakan.
        
    -   **Relevansi:** _Coverage_ yang lebih tinggi menunjukkan bahwa sistem mampu merekomendasikan beragam item dari katalog, bukan hanya sebagian kecil item populer. Ini penting untuk penemuan item dan menghindari filter gelembung. Namun, _coverage_ yang sangat tinggi tanpa akurasi juga tidak diinginkan.

### 2. Metrik Evaluasi untuk Collaborative Filtering

-   **Root Mean Squared Error (RMSE):**
    -   **Penjelasan:** RMSE mengukur rata-rata magnitudo kesalahan antara nilai prediksi dan nilai aktual. Dalam kasus ini, ini adalah perbedaan antara _playcount_ yang dinormalisasi yang diprediksi oleh model dan _playcount_ yang dinormalisasi aktual dari data validasi.
    -   **Formula:**
        $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
        Di mana:
        -   $N$ adalah jumlah total observasi (interaksi pengguna-item).
        -   $y_i$ adalah nilai aktual (misalnya, _playcount_ ternormalisasi) untuk observasi ke-i.
        -   $\hat{y}_i$ adalah nilai prediksi oleh model untuk observasi ke-i.
    -   **Cara Kerja:** Untuk setiap pasangan pengguna-item dalam set validasi, model memprediksi skor preferensi (nilai antara 0 dan 1). RMSE kemudian menghitung akar dari rata-rata kuadrat perbedaan antara prediksi ini dan nilai _playcount_ ternormalisasi yang sebenarnya. Nilai RMSE yang lebih rendah menunjukkan kinerja model yang lebih baik, artinya prediksi model lebih dekat dengan nilai aktual.
    -   **Hasil (Berdasarkan Kode):**
        Hasil RMSE untuk data latih (`root_mean_squared_error`) dan data validasi (`val_root_mean_squared_error`) ditampilkan selama proses pelatihan model. Plot _history_ pelatihan menunjukkan bagaimana metrik ini berubah sepanjang epoch.
        ```python
        plt.figure(figsize=(10, 6)) #
        plt.plot(history.history['root_mean_squared_error'], label='train_rmse') #
        plt.plot(history.history['val_root_mean_squared_error'], label='val_rmse') #
        plt.title('Model Metrics (Collaborative Filtering)') 
        plt.ylabel('root_mean_squared_error') 
        plt.xlabel('epoch') 
        plt.legend(loc='upper left') 
        plt.grid(True) 
        plt.show() 
        ```

        ![train-plot](https://github.com/user-attachments/assets/ef3e7251-bc7f-408f-9077-58b8ec350916)
        
        Nilai `root_mean_squared_error` dan `val_root_mean_squared_error` yang dicapai 0.1586 dan 0.283 adalah indikator utama kinerja model pada data yang tidak terlihat.

    -   **Relevansi:** RMSE adalah metrik standar untuk mengevaluasi akurasi prediksi dalam sistem rekomendasi yang memprediksi rating atau skor preferensi. Ini memberikan ukuran seberapa baik model dapat memperkirakan interaksi pengguna-item.

## Kesimpulan

Proyek ini berhasil mengembangkan dua jenis sistem rekomendasi musik: _Content-Based Filtering_ dan _Collaborative Filtering_, dan mengevaluasinya menggunakan metrik yang sesuai.

1.  **Content-Based Filtering**, yang menggunakan TF-IDF pada genre musik dan _cosine similarity_, mampu menghasilkan rekomendasi lagu yang memiliki kemiripan genre. Evaluasi menggunakan metrik _Coverage_ menunjukkan kemampuan sistem untuk merekomendasikan sebagian dari katalog lagu yang tersedia.
2.  **Collaborative Filtering**, diimplementasikan menggunakan model _neural network_ dengan _embedding layers_, belajar dari pola interaksi pengguna (jumlah putar). Kinerja model dievaluasi menggunakan RMSE pada data validasi, yang menunjukkan seberapa akurat model dapat memprediksi preferensi pengguna terhadap lagu. Nilai RMSE yang dicapai 0.283 pada data validasi menunjukkan bahwa model memiliki kemampuan prediktif yang cukup baik dalam konteks data yang dinormalisasi.
3.  Kedua pendekatan memiliki kelebihan dan kekurangan masing-masing. _Content-Based_ baik untuk _cold start_ item dan transparansi, sementara _Collaborative Filtering_ dapat menemukan rekomendasi yang lebih beragam (_serendipitous_) tetapi mempunyai masalah _cold start_ pengguna/item dan membutuhkan banyak data interaksi.
4.  Proses _Data Understanding_ dan _Data Preparation_ sangat krusial. Langkah-langkah seperti pembersihan genre, normalisasi _playcount_, dan encoding ID sangat penting untuk keberhasilan kedua model.

Proyek ini menunjukkan bagaimana teknik _machine learning_ dapat diterapkan untuk membangun sistem rekomendasi musik yang fungsional. Pengembangan lebih lanjut dapat mencakup penggabungan kedua pendekatan menjadi sistem _hybrid_, eksplorasi fitur konten yang lebih kaya (misalnya, fitur audio), dan _tuning hyperparameter_ yang lebih ekstensif untuk kedua model.

## Referensi

1.  Gao, C., Zheng, Y., Li, N., Li, Y., Qin, Y., Piao, J., Quan, Y., Chang, J., & Jin, D. (2022). A Survey of Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions. ACM Transactions on Recommender Systems, 1(1), Article 3. [https://dl.acm.org/doi/pdf/10.1145/3568022](https://dl.acm.org/doi/pdf/10.1145/3568022)
2.  He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering. arXiv preprint arXiv:1708.05031. [https://arxiv.org/pdf/1708.05031](https://arxiv.org/pdf/1708.05031) 
3.  Dicoding Indonesia. (n.d.).  _Machine Learning Terapan_. Dicoding Academy. Diakses pada 31 Mei 2025, dari https://www.dicoding.com/academies/319-machine-learning-terapan
