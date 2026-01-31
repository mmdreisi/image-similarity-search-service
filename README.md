# CheckThem — Visual Template Search for Websites 🔎🖼️

**CheckThem** is an open-source project for searching website templates by image. Upload an image, extract its features using DINOv2, and use FAISS to find the most visually similar templates from a template database.

---

## ✨ Key Features

- **Image-based search** (Upload or Drag & Drop) ✅
- Feature extraction using DINOv2 (pretrained or finetuned) 🤖
- Fast vector storage and nearest-neighbor search with **FAISS** ⚡
- Simple API built with **FastAPI** (endpoints: `/api/search_image`, `/api/save_image`) 🔗
- Quick frontend pages: `upload.html` and `results.html` for testing 🧭

---

## Project Structure

- `api.py` — FastAPI server that handles search requests and background image saving
- `features/extractor_dino_v2.py` — DINOv2-based feature extractor
- `db/vector_store.py` — FAISS wrapper for adding, saving and searching vectors
- `db/read_sql.py` — utility to read templates from a local SQLite DB (used by `add_demo.py`)
- `add_demo.py` — example script to build a vector index from the templates DB
- `upload.html`, `results.html` — simple frontend to upload images and view results
- Data and index files:
  - `data/upload/images/` — uploaded images served by the API
  - `data/vector_data/vector.index` — FAISS index file
  - `data/vector_data/meta.pkl` — pickled metadata for indexed items

---

## Requirements

- Python 3.8+ (recommended: 3.10/3.11)
- GPU is optional but recommended for faster feature extraction with larger models

The existing `requierments.txt` contains the core dependencies, but for development and running you will likely need:

```bash
pip install -r requierments.txt
pip install transformers pillow faiss-cpu numpy pandas uvicorn python-multipart
```

If you have a CUDA-capable GPU and want to use FAISS with GPU support, install `faiss-gpu` instead of `faiss-cpu`.

---

## Quick Start (Local)

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies (see the Requirements section).

3. (Optional) Build the vector index from your templates DB by setting the `db_path` in `add_demo.py` and running:

```bash
python add_demo.py
# This will create/update data/vector_data/vector.index and data/vector_data/meta.pkl
```

4. Start the API server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

5. Open `http://localhost:8000/upload.html` in your browser to upload a test image.

---

## API Usage

- POST `/api/search_image` — upload an image file (form-data key: `image`) to search for similar templates
  - Response: a list of results (metadata + similarity percentage)

- POST `/api/save_image` — enqueue an image for asynchronous saving (non-blocking)

- Static pages:
  - `/upload.html` — upload UI
  - `/results.html` — displays the search results pulled from `sessionStorage`

---

## Development Notes & Configuration

- Feature extraction is implemented in `features/extractor_dino_v2.py`. You can change `model_name` or pass a `finetune_path` to load finetuned weights.
- `db/vector_store.py` uses `IndexFlatIP` and converts cosine similarity to a percentage for display.
- `NUM_WORKERS` in `api.py` controls how many background workers will save images from the queue.

---

## Examples / Usage Scenarios

- To add new templates to the vector DB:
  1. Add records to the SQLite DB (or update the path in `add_demo.py`).
  2. Run `python add_demo.py` to rebuild/update the FAISS index.

- Use `results.html` to inspect returned matches visually; the frontend stores the search response in `sessionStorage` and renders it there.

---

## Contributing 🤝

Bug reports, feature requests and pull requests are welcome. Please add tests where appropriate and update `requierments.txt` if you introduce new dependencies.
