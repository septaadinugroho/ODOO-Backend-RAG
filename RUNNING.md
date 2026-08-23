# Panduan Menjalankan Class Bot

Panduan ini menjelaskan cara menjalankan servis Class Bot dari nol sampai bisa
menerima query di Swagger `/docs`. Ada dua jalur: **jalur cepat** (demo tanpa akun
Google Cloud) dan **jalur lengkap** (pakai Vertex AI beneran). Semua perintah
dijalankan dari folder root proyek (`ODOO-backend-RAG`).

Kalau kamu cuma mau paham apa itu servis ini dan di mana tiap requirement PRD
diimplementasikan, baca **[README.md](README.md)** dulu.


## Prasyarat

Yang perlu ada di komputer kamu sebelum mulai:

- **Python 3.10 atau lebih baru** — cek dengan `python3 --version`.
- **Docker + Docker Compose** — buat menjalankan PostgreSQL 16. Kalau nggak mau
  pakai Docker, kamu bisa pakai PostgreSQL sendiri (lihat bagian *Alternatif tanpa
  Docker* di Langkah 3).
- **(Opsional) Akun Google Cloud** — cuma diperlukan kalau mau LLM beneran (state
  `answered` dengan narasi). Buat demo, kita pakai mode `null` yang nggak butuh ini.

> Catatan: `pytest` **tidak** butuh Docker maupun Google Cloud. Kalau kamu cuma mau
> menjalankan test, lompat langsung ke bagian *Menjalankan test*.


## Jalur cepat (demo tanpa Google Cloud)

Empat langkah, dari nol sampai bisa nge-query di browser:

```bash
./setup.sh                              # 1. bikin .venv + install semua dependency
# edit .env: ganti LLM_PROVIDER=vertex jadi LLM_PROVIDER=null   (2. lihat Langkah 2)
docker compose up -d                    # 3. PostgreSQL 16 di port 5433, sudah terisi corpus
source .venv/bin/activate && uvicorn app.main:app --reload --port 8000   # 4. jalankan API
```

Buka http://127.0.0.1:8000/docs, klik **Authorize**, tempel
`odoo-dev-key-change-me`, lalu coba `POST /v1/chat`. Selesai. Di mode `null`,
pencarian tetap jalan — jawabannya balik dengan `state: "degraded"` (tanpa paragraf
narasi buatan model), dan daftar kelasnya utuh.

Bagian berikutnya menjelaskan tiap langkah lebih detail, plus jalur Vertex AI.


## Langkah 1 — Pasang dependency

Cara termudah, pakai skrip:

```bash
./setup.sh
```

Skrip ini mencari Python 3.10+, membuat virtual environment di `./.venv`,
meng-install `requirements.txt` ke dalamnya, dan menyalin `.env.example` jadi `.env`
kalau `.env` belum ada (file `.env` kamu yang sudah ada tidak akan ditimpa).
Terakhir, skrip menjalankan smoke import supaya kalau ada yang salah, gagalnya di
sini — bukan pas request pertama. Aman dijalankan berulang kali.

Kalau mau manual (atau `setup.sh` tidak jalan di lingkungan kamu):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env        # cuma sekali; jangan timpa .env yang sudah kamu isi
```

Setiap buka terminal baru, aktifkan dulu virtual environment-nya:

```bash
source .venv/bin/activate
```


## Langkah 2 — Konfigurasi `.env`

`setup.sh` sudah membuat `.env` dari `.env.example`. Sebagian besar nilai default-nya
sudah pas buat lokal. Yang perlu kamu perhatikan:

**Pilih provider LLM.** Ini keputusan paling penting untuk demo:

- `LLM_PROVIDER=null` — **tanpa Google Cloud.** Pencarian jalan penuh, tapi jawaban
  selalu `state: "degraded"` (narasi pakai template, bukan model). Paling cepat buat
  nyoba. Ganti baris `LLM_PROVIDER=vertex` di `.env` jadi `LLM_PROVIDER=null`.
- `LLM_PROVIDER=vertex` — **pakai Vertex AI beneran** (default). Perlu setup Google
  Cloud dulu; lihat bagian *Pakai Vertex AI* di bawah.

**API keys (NFR-D).** Default-nya:

```
API_KEYS=odoo:odoo-dev-key-change-me,frontend:frontend-dev-key-change-me
```

Formatnya `konsumen:kunci`, dipisah koma. Odoo dan frontend **wajib** pakai kunci
berbeda. Saat pakai di Swagger/curl, yang dikirim di header `X-API-Key` adalah
**bagian kunci-nya saja** — misalnya `odoo-dev-key-change-me`. Sebelum produksi,
ganti semua kunci `*-change-me` ini dengan kunci acak yang panjang.

**Database.** `DATABASE_URL` default sudah menunjuk ke port `5433` yang dipakai
Docker Compose — jadi kalau kamu pakai Docker (Langkah 3), tidak perlu diubah.

**Corpus CSV.** Baris `CORPUS_CSV` default menunjuk ke file 200-record yang **belum
ada** di repo. Kalau kamu menjalankan skrip loader (`scripts/load_data.py`), arahkan
ke fixture 12-record yang tersedia:

```
CORPUS_CSV=data/sample_corpus.csv
```

Kalau kamu pakai Docker, ini tidak wajib diubah — database Docker sudah terisi
corpus sample secara otomatis (lihat Langkah 3).


## Langkah 3 — Nyalakan PostgreSQL

```bash
docker compose up -d
```

Perintah ini menjalankan PostgreSQL 16 di **port 5433** (5432 sengaja dihindari
karena dipakai proyek `python-backend-RAG`). Pertama kali dijalankan di atas volume
kosong, container otomatis menjalankan `db/init/001_schema.sql` dan
`002_seed_corpus.sql` — jadi begitu container sehat, **database sudah punya skema,
kosakata tertutup, dan 12 record corpus, tanpa perlu menjalankan skrip loader apa
pun.** Healthcheck-nya sengaja ketat: container baru dilaporkan `healthy` setelah
tabel `class_records` benar-benar terisi, jadi apa pun yang menunggu container ini
tidak akan balapan dengan proses seeding.

Cek statusnya:

```bash
docker compose ps                 # tunggu sampai kolom STATUS = healthy
```

Kalau kamu perlu membangun ulang database dari awal (misalnya mengubah SQL di
`db/init/`):

```bash
docker compose down -v && docker compose up -d      # -v menghapus volume data
```

### Alternatif tanpa Docker

Kalau kamu sudah punya PostgreSQL sendiri, arahkan `DATABASE_URL` di `.env` ke
instance itu, lalu buat skema dan muat corpus lewat skrip (butuh `.venv` aktif):

```bash
python scripts/init_db.py                 # buat tabel + seed kosakata tertutup
python scripts/init_db.py --check         # cek koneksi + jumlah baris saja
python scripts/load_data.py data/sample_corpus.csv --replace   # muat corpus
```

`load_data.py` menerima `--dry-run` (validasi tanpa menulis) dan `--replace`
(hapus `class_records` + `providers` lama dulu). Semua penulisan pakai *parameterized
query*, jadi aman dari SQL injection.


## Langkah 4 — Jalankan API

```bash
source .venv/bin/activate        # kalau belum aktif
uvicorn app.main:app --reload --port 8000
```

Servis sekarang jalan di http://127.0.0.1:8000. Endpoint yang tersedia:

- `POST /v1/chat` — endpoint utama pencarian kelas (butuh `X-API-Key`).
- `GET /health` — status servis, corpus, dan provider LLM (publik, tanpa kunci).
- `GET /metrics` — latency, jumlah request per state, rasio degraded, biaya/token
  (butuh `X-API-Key`).
- `GET /docs` — Swagger UI interaktif.

Cek cepat bahwa servis hidup (tidak perlu kunci):

```bash
curl http://127.0.0.1:8000/health
```

Kalau pakai mode `null`, `status`-nya akan `degraded` dan `llm_ready` bernilai
`false` — itu memang perilaku yang benar, bukan error.


## Langkah 5 — Demo di Swagger `/docs`

1. Buka http://127.0.0.1:8000/docs.
2. Klik tombol **Authorize** di kanan atas.
3. Tempel nilai kunci — misalnya `odoo-dev-key-change-me` — lalu **Authorize**,
   **Close**.
4. Buka `POST /v1/chat`, klik **Try it out**, ganti isi `message`, lalu **Execute**.

Contoh pakai `curl` yang menampilkan tiap state:

```bash
KEY="odoo-dev-key-change-me"
URL="http://127.0.0.1:8000/v1/chat"

# answered / degraded — query lengkap yang cocok
curl -s -X POST "$URL" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"message":"kelas Additional Mathematics Form 4 di Selangor"}'

# needs_clarification — terlalu luas, ditanya balik satu field
curl -s -X POST "$URL" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"message":"Saya nak cari kelas tuisyen"}'

# unavailable_field — tanya harga; kelasnya tetap dikembalikan, harga tidak
curl -s -X POST "$URL" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"message":"Berapa harga kelas Additional Mathematics Form 4 Selangor?"}'

# out_of_scope — bukan pertanyaan pencarian kelas
curl -s -X POST "$URL" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"message":"What is the capital of France?"}'
```

Untuk melanjutkan percakapan multi-turn, kirim balik `session_id` dari respons
sebelumnya:

```bash
curl -s -X POST "$URL" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"message":"Form 5 sahaja","session_id":"s_demo_1"}'
```

Konteks bertahan 30 menit tanpa aktivitas (FR-28).


## Menjalankan test

Test suite berjalan di atas SQLite in-memory dengan `LLM_PROVIDER=null`, jadi
**tidak butuh Docker, tidak butuh database, tidak butuh kredensial**. Cukup:

```bash
source .venv/bin/activate
pytest
```

Menjalankan sebagian:

```bash
pytest tests/test_app.py            # kontrak endpoint: enam state, auth, error
pytest tests/test_honesty.py        # guardrail kejujuran data (FR-16/17/18)
pytest tests/test_sql_injection.py  # ketahanan terhadap SQL injection
```

Syarat **coverage 100%** untuk FR-16/17/18 diukur pada modul guardrail:

```bash
pytest --cov=app.core.honesty --cov-report=term-missing --cov-fail-under=100 \
       tests/test_honesty.py tests/test_app.py
```

Satu test — `tests/test_schema_matches_orm.py` — membandingkan skema Postgres yang
hidup dengan model SQLAlchemy. Ini satu-satunya test yang butuh Postgres, dan akan
**di-skip otomatis** kalau Postgres tidak ada. Untuk menjalankannya, nyalakan dulu
database (`docker compose up -d`) lalu:

```bash
TEST_DATABASE_URL=postgresql+psycopg2://classbot:classbot@127.0.0.1:5433/classbot \
  pytest tests/test_schema_matches_orm.py
```


## Menjalankan evaluasi (M1–M9, FR-31)

`scripts/run_eval.py` menjalankan aplikasi asli lewat `TestClient` dan mencetak tabel
metrik M1–M9. Akurasi field dan presisi filter dicek dengan membaca ulang tiap
record langsung dari database pakai parameterized query, jadi jawaban dinilai
terhadap data, bukan terhadap dirinya sendiri.

```bash
python scripts/run_eval.py                 # pakai provider dari .env
python scripts/run_eval.py --llm null      # metrik retrieval saja, tanpa token
python scripts/run_eval.py --json out.json --md out.md    # simpan laporan
python scripts/run_eval.py --metric M4     # jalankan kasus untuk satu metrik saja
```

Mode `--llm null` berguna untuk memeriksa seluruh metrik retrieval tanpa
menghabiskan kuota model.


## Pakai Vertex AI (LLM beneran)

Kalau mau state `answered` dengan narasi buatan model, bukan `degraded`:

1. Set `LLM_PROVIDER=vertex` di `.env` (ini default-nya).
2. Isi `GOOGLE_CLOUD_PROJECT` dengan ID proyek GCP kamu. Region default
   `asia-southeast1`, model default `gemini-2.0-flash` — ubah kalau perlu.
3. Login pakai Application Default Credentials:

   ```bash
   gcloud auth application-default login
   gcloud config set project <id-proyek-kamu>
   ```

4. Restart uvicorn. Cek `GET /health` — `llm_ready` harus `true` dan `status` `ok`.

Biaya per query dipantau (NFR-E, batas default RM0.05) dan muncul di `GET /metrics`.


## Troubleshooting

**Port 5433 sudah dipakai.** Ubah pemetaan port di `docker-compose.yml` (misalnya
`"5434:5432"`) dan samakan `DATABASE_URL` di `.env`.

**`docker compose ps` menunjukkan `health: starting` lama.** Wajar saat pertama
kali — image menjalankan file seed dulu dan baru lapor `healthy` setelah corpus
masuk. Tunggu sampai `start_period` (30 detik) lewat.

**Dapat `401 unauthorized`.** Header `X-API-Key` kosong atau salah. Pakai bagian
kunci saja (contoh `odoo-dev-key-change-me`), bukan `odoo:odoo-dev-key-change-me`.

**Semua jawaban `degraded` padahal mau `answered`.** Berarti LLM tidak aktif:
entah `LLM_PROVIDER=null`, atau `vertex` tapi kredensial GCP belum di-set. Cek
`GET /health` → `llm_ready`.

**`pytest` gagal dengan `ModuleNotFoundError`.** Virtual environment belum aktif atau
dependency belum ter-install. Jalankan `source .venv/bin/activate` lalu
`pip install -r requirements.txt` (atau `./setup.sh` sekali lagi).

**Error soal `CORPUS_CSV` tidak ditemukan saat `load_data.py`.** Default `.env`
menunjuk ke file 200-record yang belum ada. Set `CORPUS_CSV=data/sample_corpus.csv`,
atau berikan path CSV langsung sebagai argumen: `python scripts/load_data.py
data/sample_corpus.csv`.

**Servis menolak boot di produksi.** Saat `APP_ENV=production`, servis sengaja
menolak start kalau masih pakai kunci `*-change-me`, kunci tunggal untuk semua
konsumen, atau HTTPS dimatikan. Ganti kunci, beri kunci berbeda per konsumen, dan
aktifkan HTTPS.


## Ringkasan perintah

```bash
./setup.sh                                   # setup sekali: .venv + dependency + .env
source .venv/bin/activate                    # aktifkan venv di tiap terminal baru
docker compose up -d                         # Postgres 16 di 5433, sudah terisi corpus
docker compose down -v && docker compose up -d   # bangun ulang DB dari awal
uvicorn app.main:app --reload --port 8000    # jalankan API → http://127.0.0.1:8000/docs
pytest                                       # seluruh test (tanpa DB / GCP)
python scripts/run_eval.py --llm null        # evaluasi M1–M9 tanpa token
```
