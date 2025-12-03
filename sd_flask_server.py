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
# 默认使用公开模型 runwayml/stable-diffusion-v1-5
MODEL_NAME = os.environ.get("MODEL_NAME", "runwayml/stable-diffusion-v1-5")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/web")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Boot] Using device:", device)
print(f"[Boot] Loading model: {MODEL_NAME}")

# ===== 提前加载模型，只加载一次 =====
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)

# 使用 DPM++ 调度器（更稳定）
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 尝试启用 xformers（如果环境支持就会更快，不支持就忽略）
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
      "negative_prompt": "...",   # 可选
      "num_images": 1,            # 可选，默认 1，最多 2
      "height": 512,              # 可选，默认 512
      "width": 512,               # 可选，默认 512
      "guidance_scale": 7.5,      # 可选
      "num_inference_steps": 20,  # 可选，默认 20
      "seed": 123456              # 可选
    }
    """
    data = request.get_json(force=True)

    prompt: str = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    negative_prompt: Optional[str] = data.get("negative_prompt")

    # 免费 Render CPU 版本：默认 1 张，最多 2 张
    num_images: int = int(data.get("num_images", 1))
    num_images = max(1, min(num_images, 2))

    # 分辨率默认 512x512，并限制在 256~768 之间，避免太大
    height: int = int(data.get("height", 512))
    width: int = int(data.get("width", 512))
    height = max(256, min(height, 768))
    width = max(256, min(width, 768))

    guidance_scale: float = float(data.get("guidance_scale", 7.5))

    # 步数默认 20，限制在 10~30，CPU 再多会太慢
    num_inference_steps: int = int(data.get("num_inference_steps", 20))
    num_inference_steps = max(10, min(num_inference_steps, 30))

    seed_in = data.get("seed")
    base_seed = int(seed_in) if seed_in is not None else np.random.randint(0, 2**32)

    print(
        f"[Generate] prompt={prompt!r}, n={num_images}, "
        f"size={width}x{height}, steps={num_inference_steps}, seed={base_seed}"
    )

    generators = [
        torch.Generator(device=device).manual_seed(base_seed + i)
        for i in range(num_images)
    ]

    # 实际调用 Stable Diffusion
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
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
        }
    )


# 本地调试用，Render 上会用 gunicorn 启动
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
