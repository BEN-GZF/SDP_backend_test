from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
from flask_cors import CORS
import os


ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
MAX_MB = int(os.environ.get("MAX_MB", "30"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024  # 限制上传文件体积
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)


CORS(app, resources={r"/*": {"origins": [o.strip() for o in ALLOWED_ORIGINS.split(",")]}})

ALLOWED_MIMES = {"image/png", "image/jpeg", "image/webp"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400

    if f.mimetype not in ALLOWED_MIMES:
        return jsonify({"error": f"unsupported type: {f.mimetype}"}), 415

    name = secure_filename(f.filename)
    save_path = UPLOAD_DIR / name
    f.save(save_path)

    # TODO: 这里接入你的模型 / 队列 / HPC；返回真实的结果
    # 现在先回显并提供一个可访问的静态文件 URL（仅演示）
    return jsonify({
        "filename": name,
        "size": save_path.stat().st_size,
        "message": "received",
        "preview_url": f"/files/{name}"  # 用于前端预览
    }), 200

# 静态访问上传的临时文件（仅演示用途）
@app.get("/files/<path:filename>")
def files(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)
