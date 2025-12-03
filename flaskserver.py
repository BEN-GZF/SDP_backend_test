import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ===== 配置 =====
MODEL_NAME = os.environ.get("MODEL_NAME", "stabilityai/stable-diffusion-2-1")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/web")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Boot] Using device:", device)

# ===== 提前加载模型，只加载一次 =====
print(f"[Boot] Loading model: {MODEL_NAME}")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[Boot] xformers enabled")
except Exception as e:
    print("[Boot] xformers not enabled:", e)

pipe = pipe.to(device)

# ===== Flask 应用 =====
app = Flask(__name__)
CORS(app)  # 开发阶段允许所有域名，之后可以收紧


@app.route("/health")
def health():
    return {"status": "ok", "device": str(device)}, 200


@app.route("/images/<path:filename>")
def serve_image(filename):
    """给前端访问图片用，比如 https://xxx.onrender.com/images/xxx.png"""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/generate", methods=["POST"])
def generate():
    """
    接收 JSON:
    {
      "prompt": "...",
      "negative_prompt": "...",  # 可选
      "num_images": 4,           # 可选
      "height": 768,             # 可选
      "width": 768,              # 可选
      "guidance_scale": 7.5,     # 可选
      "num_inference_steps": 30, # 可选
      "seed": 123456             # 可选
    }
    """
    data = request.get_json(force=True)

    prompt: str = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    negative_prompt: Optional[str] = data.get("negative_prompt")
    num_images: int = int(data.get("num_images", 4))
    num_images = max(1, min(num_images, 4))  # Render 免费/低配机器别一次搞太多

    height: int = int(data.get("height", 768))
    width: int = int(data.get("width", 768))
    guidance_scale: float = float(data.get("guidance_scale", 7.5))
    num_inference_steps: int = int(data.get("num_inference_steps", 30))

    seed_in = data.get("seed")
    base_seed = int(seed_in) if seed_in is not None else np.random.randint(0, 2**32)

    print(f"[Generate] prompt={prompt}, n={num_images}, seed={base_seed}")

    generators = [
        torch.Generator(device=device).manual_seed(base_seed + i)
        for i in range(num_images)
    ]

    out = pipe(
        [prompt] * num_images,
        negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generators,
    )

    images = out.images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    urls: List[str] = []

    for idx, img in enumerate(images):
        filename = f"{timestamp}_seed-{base_seed+idx}_{idx+1}of{num_images}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        img.save(filepath)
        urls.append(f"/images/{filename}")

    return jsonify(
        {
            "images": urls,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": base_seed,
        }
    )


# 本地调试用，Render 上会用 gunicorn 启动
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
