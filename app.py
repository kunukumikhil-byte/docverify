from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from io import BytesIO
import os
import re
import hashlib
import difflib
import tempfile
import zipfile
from functools import lru_cache

from PIL import Image, ImageOps
import fitz  # PyMuPDF
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {
    "pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff",
    "txt", "docx", "pptx", "xlsx"
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["JSON_SORT_KEYS"] = False


def error(message, status=400):
    return jsonify({"ok": False, "error": message}), status


@lru_cache(maxsize=1)
def load_detector_bundle():
    """Load the trained artifact once per warm serverless instance."""
    import joblib
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ai_detector.joblib")
    if not os.path.isfile(model_path):
        return None
    return joblib.load(model_path)


def get_upload(field="file", required=True):
    uploaded = request.files.get(field)
    if not uploaded or not uploaded.filename:
        if required:
            raise ValueError("Choose a file first.")
        return None
    name = secure_filename(uploaded.filename)
    if not name:
        raise ValueError("The filename is invalid.")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Supported: PDF, images, TXT, DOCX, PPTX, XLSX.")
    data = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ValueError("The selected file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File is too large. Maximum size is {MAX_UPLOAD_MB} MB.")
    return name, ext, data


def extract_text(name, ext, data):
    """Extract text from supported files. OCR is optional and handled separately."""
    try:
        if ext == "txt":
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    return data.decode(encoding)
                except UnicodeDecodeError:
                    pass
            return data.decode("utf-8", errors="replace")
        if ext == "pdf":
            with fitz.open(stream=data, filetype="pdf") as pdf:
                return "\n".join(page.get_text("text") for page in pdf)
        if ext == "docx":
            doc = Document(BytesIO(data))
            chunks = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    chunks.append(" | ".join(cell.text for cell in row.cells))
            return "\n".join(chunks)
        if ext == "pptx":
            prs = Presentation(BytesIO(data))
            return "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text_frame"))
        if ext == "xlsx":
            wb = load_workbook(BytesIO(data), read_only=True, data_only=True)
            rows = []
            for ws in wb.worksheets:
                rows.append(f"[Sheet: {ws.title}]")
                for row in ws.iter_rows(values_only=True):
                    rows.append(" | ".join("" if v is None else str(v) for v in row))
            wb.close()
            return "\n".join(rows)
        if ext in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}:
            return ""
    except Exception as exc:
        raise ValueError("Could not read this file. It may be corrupted or password-protected.") from exc
    return ""


def normalized_words(text):
    return re.findall(r"\b[\w'-]+\b", (text or "").lower(), flags=re.UNICODE)


def compare_texts(text_a, text_b):
    a_words, b_words = normalized_words(text_a), normalized_words(text_b)
    if not a_words or not b_words:
        return {"similarity_percent": 0.0, "matching_passages": [], "note": "One or both documents contain no extractable text."}
    matcher = difflib.SequenceMatcher(None, a_words, b_words, autojunk=False)
    matching = []
    for block in matcher.get_matching_blocks():
        if block.size >= 5:
            phrase = " ".join(a_words[block.a:block.a + block.size])
            matching.append({"words": block.size, "text": phrase[:800]})
    # Symmetric containment-style score: matching words over the larger document.
    matching_words = sum(item["words"] for item in matching)
    score = min(100.0, 200.0 * matching_words / max(1, len(a_words) + len(b_words)))
    return {
        "similarity_percent": round(score, 2),
        "matching_passages": matching[:30],
        "word_count_a": len(a_words),
        "word_count_b": len(b_words),
        "note": "This compares only the supplied texts. It does not search the internet or establish plagiarism by itself."
    }


def make_pdf_from_images(image_bytes_list):
    out = BytesIO()
    pdf = canvas.Canvas(out)
    for raw in image_bytes_list:
        try:
            image = Image.open(BytesIO(raw))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as exc:
            raise ValueError("One of the images is invalid or corrupted.") from exc
        width, height = image.size
        page_w, page_h = 595.27, 841.89  # A4 points
        scale = min(page_w / width, page_h / height)
        draw_w, draw_h = width * scale, height * scale
        pdf.setPageSize((page_w, page_h))
        pdf.drawImage(ImageReader(image), (page_w - draw_w) / 2, (page_h - draw_h) / 2,
                      width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
        pdf.showPage()
    pdf.save()
    out.seek(0)
    return out


@app.get("/")
def home():
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "DocVerify", "max_upload_mb": MAX_UPLOAD_MB})


@app.post("/api/extract")
def api_extract():
    try:
        name, ext, data = get_upload()
        text = extract_text(name, ext, data)
        return jsonify({"ok": True, "filename": name, "text": text, "characters": len(text)})
    except ValueError as exc:
        return error(str(exc))
    except Exception:
        app.logger.exception("Unexpected extraction error")
        return error("Unable to process this file.", 500)


@app.post("/api/plagiarism/compare")
def plagiarism_compare():
    try:
        a_name, a_ext, a_data = get_upload("file1")
        b_name, b_ext, b_data = get_upload("file2")
        a_text = extract_text(a_name, a_ext, a_data)
        b_text = extract_text(b_name, b_ext, b_data)
        if not a_text.strip() or not b_text.strip():
            return error("Both files need extractable text. Image-only PDFs require OCR first.")
        result = compare_texts(a_text, b_text)
        result.update({"ok": True, "file1": a_name, "file2": b_name})
        return jsonify(result)
    except ValueError as exc:
        return error(str(exc))
    except Exception:
        app.logger.exception("Unexpected comparison error")
        return error("Unable to compare these files.", 500)


@app.post("/api/plagiarism/text")
def plagiarism_text():
    payload = request.get_json(silent=True) or {}
    text_a = payload.get("text1", "")
    text_b = payload.get("text2", "")
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return error("Both text fields must be strings.")
    if not text_a.strip() or not text_b.strip():
        return error("Enter text in both fields.")
    if len(text_a) > 200_000 or len(text_b) > 200_000:
        return error("Each text field must be 200,000 characters or fewer.")
    result = compare_texts(text_a, text_b)
    result["ok"] = True
    return jsonify(result)


@app.post("/api/ai-detect")
def ai_detect():
    """Run the locally trained classifier. No hosted inference API or token is used."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not isinstance(text, str):
        return error("Text must be a string.")
    text = text.strip()
    if len(text) < 100:
        return error("Enter at least 100 characters. Longer samples (500+ characters) usually provide a more useful estimate.")
    if len(text) > 20_000:
        return error("Text is too long. Maximum is 20,000 characters per check.")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ai_detector.joblib")
    if not os.path.isfile(model_path):
        return jsonify({
            "ok": False,
            "error": "The trained model file is missing. Train it in Google Colab using train_ai_detector.py, then place models/ai_detector.joblib in this project and redeploy."
        }), 503

    try:
        bundle = load_detector_bundle()
        if bundle is None:
            return jsonify({"ok": False, "error": "The trained model file is missing. Train it in Google Colab using train_ai_detector.py, then place models/ai_detector.joblib in this project and redeploy."}), 503
        pipeline = bundle["pipeline"] if isinstance(bundle, dict) else bundle
        metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
        probabilities = pipeline.predict_proba([text])[0]
        classes = list(pipeline.classes_)
        ai_index = classes.index(1) if 1 in classes else classes.index("ai") if "ai" in classes else None
        if ai_index is None:
            return error("The model artifact has an unexpected label mapping. Retrain using the included training script.", 500)
        ai_score = float(probabilities[ai_index])
        human_score = 1.0 - ai_score
        if ai_score >= 0.65:
            verdict = "More AI-like signals"
        elif ai_score <= 0.35:
            verdict = "More human-like signals"
        else:
            verdict = "Uncertain / mixed signals"
        return jsonify({
            "ok": True,
            "ai_likelihood_percent": round(ai_score * 100, 2),
            "human_likelihood_percent": round(human_score * 100, 2),
            "verdict": verdict,
            "model": metadata.get("model_name", "DocVerify TF-IDF + Logistic Regression"),
            "training_samples": metadata.get("training_samples"),
            "note": "This is an experimental model estimate, not proof of authorship. Performance depends on the training data and may be wrong for edited AI text, short text, new models, non-standard English, or writing styles unlike the training set. Do not use it alone for disciplinary or high-stakes decisions."
        })
    except Exception:
        app.logger.exception("Local AI detector failed to load or predict")
        return error("The trained detector could not run. Verify that models/ai_detector.joblib was created by the included training script and that runtime dependencies are installed.", 500)


@app.post("/api/convert/images-to-pdf")
def images_to_pdf():
    uploads = request.files.getlist("files")
    if not uploads or all(not f.filename for f in uploads):
        return error("Choose one or more images.")
    if len(uploads) > 30:
        return error("Upload at most 30 images per PDF.")
    raws = []
    for f in uploads:
        if not f.filename:
            continue
        name = secure_filename(f.filename)
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}:
            return error(f"{name or 'File'} is not a supported image.")
        data = f.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            return error(f"{name} is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            return error(f"{name} exceeds the {MAX_UPLOAD_MB} MB per-file limit.")
        raws.append(data)
    if not raws:
        return error("Choose one or more images.")
    try:
        result = make_pdf_from_images(raws)
        return send_file(result, mimetype="application/pdf", as_attachment=True, download_name="docverify-scan.pdf")
    except ValueError as exc:
        return error(str(exc))
    except Exception:
        app.logger.exception("Image conversion failed")
        return error("Could not convert these images to PDF.", 500)


@app.post("/api/convert/pdf-to-images")
def pdf_to_images():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return error("Choose a PDF first.")
    name = secure_filename(uploaded.filename)
    if not name.lower().endswith(".pdf"):
        return error("Upload a PDF file.")
    data = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        return error("The selected file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        return error(f"File is too large. Maximum size is {MAX_UPLOAD_MB} MB.")
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if len(pdf) > 50:
                return error("PDF has more than 50 pages. Split it before converting.")
            memory = BytesIO()
            with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
                for index, page in enumerate(pdf):
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    archive.writestr(f"page-{index + 1}.png", pix.tobytes("png"))
            memory.seek(0)
        return send_file(memory, mimetype="application/zip", as_attachment=True, download_name="docverify-pages.zip")
    except Exception as exc:
        if isinstance(exc, ValueError):
            return error(str(exc))
        app.logger.exception("PDF conversion failed")
        return error("Could not read this PDF. It may be corrupted or encrypted.")


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return error(f"Request is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.", 413)


@app.errorhandler(404)
def not_found(_exc):
    if request.path.startswith("/api/"):
        return error("API endpoint not found.", 404)
    return "Page not found", 404


@app.errorhandler(405)
def method_not_allowed(_exc):
    if request.path.startswith("/api/"):
        return error("Method not allowed for this endpoint.", 405)
    return "Method not allowed", 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
