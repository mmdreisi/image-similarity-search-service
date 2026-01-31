from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from features.extractor_dino_v2 import FeatureExtractor
from db.vector_store import VectorStore
from PIL import Image
from contextlib import asynccontextmanager
import logging
import asyncio
import io
import uuid
import os

# -----------------------------
# تنظیمات اولیه
# -----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

upload_image_dir = Path("data/upload/images")
upload_image_dir.mkdir(parents=True, exist_ok=True)

extractor = FeatureExtractor()
vector_store = VectorStore()

# صف ذخیره‌سازی تصاویر
save_queue = asyncio.Queue()
NUM_WORKERS = 10  # تعداد workerهای همزمان


# -----------------------------
# توابع کمکی
# -----------------------------
async def save_image_async(upload_image_path: Path, contents: bytes):
    """ذخیره غیربلاک‌کننده‌ی تصویر روی دیسک"""
    loop = asyncio.get_event_loop()

    def write_file():
        with open(upload_image_path, "wb") as f:
            f.write(contents)

    await loop.run_in_executor(None, write_file)
    logger.debug(f"✅ Image saved successfully at: {upload_image_path}")


async def image_saver_worker(worker_id: int):
    """Worker پس‌زمینه برای ذخیره‌سازی صف تصاویر"""
    logger.info(f"🧵 Worker {worker_id} started.")
    while True:
        upload_image_path, contents = await save_queue.get()
        try:
            await save_image_async(upload_image_path, contents)
            logger.info(f"✅ Worker {worker_id}: saved {upload_image_path.name}")
        except Exception as e:
            logger.error(f"❌ Worker {worker_id}: failed to save image - {e}")
        finally:
            save_queue.task_done()


# -----------------------------
# Lifespan handler (جایگزین on_event)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    for i in range(NUM_WORKERS):
        asyncio.create_task(image_saver_worker(i))
    logger.info(f"🚀 Started {NUM_WORKERS} image saver workers.")

    yield 

    logger.info("🧹 Shutting down workers...")


# -----------------------------
# تعریف برنامه FastAPI
# -----------------------------
app = FastAPI(
    title="CheckThem API",
    description="API for finding similar website templates",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=upload_image_dir), name="uploads")


# -----------------------------
# اندپوینت اصلی جستجوی تصویر
# -----------------------------
@app.post("/api/search_image")
async def search_image(image: UploadFile = File(...)):
    logger.info(f"📥 Received /api/search_image request with file: {image.filename}")
    try:
        contents = await image.read()

        if image.content_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.debug("🧠 Extracting image features...")
        features = extractor.extract_image(img)

        logger.debug("🔍 Searching vector store...")
        results = vector_store.search(features, top_k=20)

        formatted_results = [
            {
                "metadata": {
                    "template_name": meta.get("template_name", "Unknown Template"),
                    "template_url": meta.get("template_url", "#"),
                    "demo_name": meta.get("demo_name", "View Demo"),
                    "demo_url": meta.get("demo_url", "#"),
                    "image_path": meta.get("image_path", "/static/placeholder.jpg"),
                },
                "similarity_percentage": float(similarity),
            }
            for meta, similarity in results
        ]

        logger.info(f"✅ Image search completed for {image.filename}")
        return {"query_filename": image.filename, "results": formatted_results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /api/search_image: {e}")
        return JSONResponse({"detail": str(e)}, status_code=500)


# -----------------------------
# اندپوینت صف‌بندی و ذخیره تصویر
# -----------------------------
@app.post("/api/save_image")
async def enqueue_image_for_saving(image: UploadFile = File(...)):
    """اضافه کردن تصویر به صف ذخیره‌سازی بدون بلاک کردن درخواست"""
    try:
        contents = await image.read()

        if image.content_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
            raise HTTPException(status_code=400, detail="Unsupported image format")

        filename = f"{uuid.uuid4()}{Path(image.filename).suffix}"
        upload_image_path = upload_image_dir / filename

        await save_queue.put((upload_image_path, contents))
        logger.info(f"🟢 Enqueued image for saving: {filename}")

        return {"status": "queued", "filename": filename}

    except Exception as e:
        logger.error(f"❌ Failed to enqueue image: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ================================================================
# 🔸 سرو کردن صفحات HTML و فایل‌های استاتیک
# ================================================================
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request

BASE_DIR = Path(__file__).parent

# ✅ سرو style.css و فایل‌های HTML
app.mount("/", StaticFiles(directory=BASE_DIR), name="root_static")

# ✅ سرو فونت‌ها و CSSهای پوشه assets
if (BASE_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=BASE_DIR / "assets"), name="assets")

# ✅ سرو تصاویر آپلود شده
upload_dir = BASE_DIR / "data" / "upload" / "images"
if upload_dir.exists():
    app.mount("/images", StaticFiles(directory=upload_dir), name="uploaded_images")

# 📄 upload.html
@app.get("/", response_class=HTMLResponse)
@app.get("/upload.html", response_class=HTMLResponse)
async def serve_upload_html(request: Request):
    return HTMLResponse(content=(BASE_DIR / "upload.html").read_text(encoding="utf-8"))

# 📄 results.html
@app.get("/results.html", response_class=HTMLResponse)
async def serve_results_html(request: Request):
    return HTMLResponse(content=(BASE_DIR / "results.html").read_text(encoding="utf-8"))
