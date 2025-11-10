import os
import re
import tempfile
import shutil
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import uvicorn
from decouple import config as env_config

from distutils.command.clean import clean
import httpx
import json
import random
import time
import io
from datetime import datetime
import requests
import asyncio
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Lifespan context for model loading
# -----------------------------
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, tokenizer
    
    # Environment setup
    os.environ.pop("TRANSFORMERS_CACHE", None)
    MODEL_NAME = env_config("MODEL_NAME", default="deepseek-ai/DeepSeek-OCR")
    HF_HOME = env_config("HF_HOME", default="/models")
    os.makedirs(HF_HOME, exist_ok=True)
    
    # Load model
    print(f"🚀 Loading {MODEL_NAME}...")
    torch_dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    ).eval().to("cuda")
    
    # Pad token setup
    try:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    
    print("✅ Model loaded and ready!")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down...")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="DeepSeek-OCR API",
    description="Blazing fast OCR with DeepSeek-OCR model 🔥",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(
    mode: str,
    user_prompt: str,
    grounding: bool,
    find_term: Optional[str],
    schema: Optional[str],
    include_caption: bool,
) -> str:
    """Build the prompt based on mode"""
    parts: List[str] = ["<image>"]
    mode_requires_grounding = mode in {"find_ref", "layout_map", "pii_redact"}
    if grounding or mode_requires_grounding:
        parts.append("<|grounding|>")

    instruction = ""
    if mode == "plain_ocr":
        instruction = "Free OCR."
    elif mode == "markdown":
        instruction = "Convert the document to markdown."
    elif mode == "tables_csv":
        instruction = (
            "Extract every table and output CSV only. "
            "Use commas, minimal quoting. If multiple tables, separate with a line containing '---'."
        )
    elif mode == "tables_md":
        instruction = "Extract every table as GitHub-flavored Markdown tables. Output only the tables."
    elif mode == "kv_json":
        schema_text = schema.strip() if schema else "{}"
        instruction = (
            "Extract key fields and return strict JSON only. "
            f"Use this schema (fill the values): {schema_text}"
        )
    elif mode == "figure_chart":
        instruction = (
            "Parse the figure. First extract any numeric series as a two-column table (x,y). "
            "Then summarize the chart in 2 sentences. Output the table, then a line '---', then the summary."
        )
    elif mode == "find_ref":
        key = (find_term or "").strip() or "Total"
        instruction = f"Locate <|ref|>{key}<|/ref|> in the image."
    elif mode == "layout_map":
        instruction = (
            'Return a JSON array of blocks with fields {"type":["title","paragraph","table","figure"],'
            '"box":[x1,y1,x2,y2]}. Do not include any text content.'
        )
    elif mode == "pii_redact":
        instruction = (
            'Find all occurrences of emails, phone numbers, postal addresses, and IBANs. '
            'Return a JSON array of objects {label, text, box:[x1,y1,x2,y2]}.'
        )
    elif mode == "multilingual":
        instruction = "Free OCR. Detect the language automatically and output in the same script."
    elif mode == "describe":
        instruction = "Describe this image. Focus on visible key elements."
    elif mode == "freeform":
        instruction = user_prompt.strip() if user_prompt else "OCR this image."
    else:
        instruction = "OCR this image."

    if include_caption and mode not in {"describe"}:
        instruction = instruction + "\nThen add a one-paragraph description of the image."

    parts.append(instruction)
    return "\n".join(parts)

# -----------------------------
# Grounding parser
# -----------------------------
# Match a full detection block and capture the coordinates as the entire list expression
# Examples of captured coords (including outer brackets):
#  - [[312, 339, 480, 681]]
#  - [[504, 700, 625, 910], [771, 570, 996, 996]]
#  - [[110, 310, 255, 800], [312, 343, 479, 680], ...]
# Using a greedy bracket capture ensures we include all inner lists up to the last ']' before </|det|>
DET_BLOCK = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>\s*(?P<coords>\[.*\])\s*<\|/det\|>",
    re.DOTALL,
)

def clean_grounding_text(text: str) -> str:
    """Remove grounding tags from text for display, keeping labels"""
    # Replace <|ref|>label<|/ref|><|det|>[...any nested lists...]<|/det|> with just the label
    cleaned = re.sub(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*\[.*\]\s*<\|/det\|>",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    # Also remove any standalone grounding tags
    cleaned = re.sub(r"<\|grounding\|>", "", cleaned)
    return cleaned.strip()

def parse_detections(text: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """Parse grounding boxes from text and scale from 0-999 normalized coords to actual image dimensions
    
    Handles both single and multiple bounding boxes:
    - Single: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
    - Multiple: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]<|/det|>
    """
    boxes: List[Dict[str, Any]] = []
    for m in DET_BLOCK.finditer(text or ""):
        label = m.group("label").strip()
        coords_str = m.group("coords").strip()

        print(f"🔍 DEBUG: Found detection for '{label}'")
        print(f"📦 Raw coords string (with brackets): {coords_str}")

        try:
            import ast

            # Parse the full bracket expression directly (handles single and multiple)
            parsed = ast.literal_eval(coords_str)

            # Normalize to a list of lists
            if (
                isinstance(parsed, list)
                and len(parsed) == 4
                and all(isinstance(n, (int, float)) for n in parsed)
            ):
                # Single box provided as [x1,y1,x2,y2]
                box_coords = [parsed]
                print("📦 Single box (flat list) detected")
            elif isinstance(parsed, list):
                box_coords = parsed
                print(f"📦 Boxes detected: {len(box_coords)}")
            else:
                raise ValueError("Unsupported coords structure")

            # Process each box
            for idx, box in enumerate(box_coords):
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1 = int(float(box[0]) / 999 * image_width)
                    y1 = int(float(box[1]) / 999 * image_height)
                    x2 = int(float(box[2]) / 999 * image_width)
                    y2 = int(float(box[3]) / 999 * image_height)
                    print(f"  Box {idx+1}: {box} → [{x1}, {y1}, {x2}, {y2}]")
                    boxes.append({"label": label, "box": [x1, y1, x2, y2]})
                else:
                    print(f"  ⚠️ Skipping invalid box: {box}")
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
            continue
    
    print(f"🎯 Total boxes parsed: {len(boxes)}")
    return boxes

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root(response: Response):
    response.headers["X-Robots-Tag"] = "noindex"
    return {"message": "DeepSeek-OCR API is running! 🚀", "docs": "/docs"}

@app.get("/health")
async def health(response: Response):
    response.headers["X-Robots-Tag"] = "noindex"
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/process-images-payload")
async def process_images_batch(response: Response):
    response.headers["X-Robots-Tag"] = "noindex"
    await start_batch()
    return {"message": "DeepSeek-OCR API batch started"}


@app.post("/api/ocr")
async def ocr_inference(
    image: UploadFile = File(...),
    mode: str = Form("plain_ocr"),
    prompt: str = Form(""),
    grounding: bool = Form(False),
    include_caption: bool = Form(False),
    find_term: Optional[str] = Form(None),
    schema: Optional[str] = Form(None),
    base_size: int = Form(1024),
    image_size: int = Form(640),
    crop_mode: bool = Form(True),
    test_compress: bool = Form(False),
):
    """
    Perform OCR inference on uploaded image
    
    - **image**: Image file to process
    - **mode**: OCR mode (plain_ocr, markdown, tables_csv, etc.)
    - **prompt**: Custom prompt for freeform mode
    - **grounding**: Enable grounding boxes
    - **include_caption**: Add image description
    - **find_term**: Term to find (for find_ref mode)
    - **schema**: JSON schema (for kv_json mode)
    - **base_size**: Base processing size
    - **image_size**: Image size parameter
    - **crop_mode**: Enable crop mode
    - **test_compress**: Test compression
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Build prompt
    prompt_text = build_prompt(
        mode=mode,
        user_prompt=prompt,
        grounding=grounding,
        find_term=find_term,
        schema=schema,
        include_caption=include_caption,
    )
    
    tmp_img = None
    out_dir = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_img = tmp.name
        
        # Get original dimensions
        try:
            with Image.open(tmp_img) as im:
                orig_w, orig_h = im.size
        except Exception:
            orig_w = orig_h = None
        
        out_dir = tempfile.mkdtemp(prefix="dsocr_")
        
        # Run inference
        res = model.infer(
            tokenizer,
            prompt=prompt_text,
            image_file=tmp_img,
            output_path=out_dir,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=False,
            test_compress=test_compress,
            eval_mode=True,
        )
        
        # Normalize response
        if isinstance(res, str):
            text = res.strip()
        elif isinstance(res, dict) and "text" in res:
            text = str(res["text"]).strip()
        elif isinstance(res, (list, tuple)):
            text = "\n".join(map(str, res)).strip()
        else:
            text = ""

        # Fallback: check output file
        if not text:
            mmd = os.path.join(out_dir, "result.mmd")
            if os.path.exists(mmd):
                with open(mmd, "r", encoding="utf-8") as fh:
                    text = fh.read().strip()
        if not text:
            text = "No text returned by model."

        # Parse grounding boxes with proper coordinate scaling
        boxes = parse_detections(text, orig_w or 1, orig_h or 1) if ("<|det|>" in text or "<|ref|>" in text) else []

        # Clean grounding tags from display text, but keep the labels
        display_text = clean_grounding_text(text) if ("<|ref|>" in text or "<|grounding|>" in text) else text

        # If display text is empty after cleaning but we have boxes, show the labels
        if not display_text and boxes:
            display_text = ", ".join([b["label"] for b in boxes])

        PRODUCT_KEYWORDS = ["orbit"]

        receipt = getReceiptData(display_text, PRODUCT_KEYWORDS)

        return JSONResponse({
            "success": True,
            "text": display_text,
            "detected": receipt,
            # "raw_text": text,  # Include raw model output for debugging
            "boxes": boxes,
            "image_dims": {"w": orig_w, "h": orig_h},
            "metadata": {
                "mode": mode,
                "grounding": grounding or (mode in {"find_ref","layout_map","pii_redact"}),
                "base_size": base_size,
                "image_size": image_size,
                "crop_mode": crop_mode
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    
    finally:
        if tmp_img:
            try:
                os.remove(tmp_img)
            except Exception:
                pass
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)

def find_products(text: str, keywords: list[str], threshold: float = 0.8):
    text_lower = text.lower()
    tokens = re.findall(r'\w+', text_lower)
    found = set()

    for word in keywords:
        for token in tokens:
            ratio = SequenceMatcher(None, word, token).ratio()
            if ratio >= threshold:
                found.add(word.capitalize())
    return list(found)

def getReceiptData(text, product_keywords=None):
    clean = re.sub(r'\s+', ' ', text.strip().replace('\n', ' '))
    res = {"fb": None, "kkm": None, "sum": None, "fallback_sum": None, "datetime": None}


    # === Product of interest ===
    if product_keywords:
        res["products_interest"] = find_products(clean, product_keywords)

    # === Дата и время ===
    mdt = re.search(
        r'(\d{2}[-.]\d{2}[-.]\d{4}|\d{2}[-.]\d{2}[-.]\d{2})[^\d]{0,3}(\d{2}:\d{2}(?::\d{2})?)',
        clean
    )

    if mdt:
        date = mdt.group(1)
        date = date.replace("-", ".")
        if len(date.split('.')[-1]) == 2:
            date = re.sub(r'(\d{2}\.\d{2}\.)(\d{2})$', r'\120\2', date)
        res["datetime"] = f"{date} {mdt.group(2)[:5]}"

    # === Фискальный признак ===
    fiscal_patterns = [
        r'\b(?:ФБ|ФП|\*\*ФП\*\*|Фискал[^\d]*|оп|oп|fp|fisc)[\s:№-]*([0-9]{6,15})',
        r'Номер\s*чека[:\s]*([0-9]{6,15})',
    ]
    for pat in fiscal_patterns:
        mf = re.search(pat, clean, re.IGNORECASE)
        if mf:
            res["fb"] = mf.group(1)
            break

    # === ККМ / РНМ / МТН ===
    mk = re.search(
        r'(?:МТН|МТН|ККМ|РНМ|РН|КГД|БКМ|Код ККМ КГД|МТН Кодш|МТН Коды|БКМ МКК МТН Коды|\*\*РНМ\*\*)'
        r'[:\s\-]*([0-9]{5,15})',
        clean,
        re.IGNORECASE
    )
    if mk:
        res["kkm"] = mk.group(1)
    else:
        fallback = re.search(r'\b(0101\d{4,11})\b', clean)
        if fallback:
            res["kkm"] = fallback.group(1)

    # === Итоговая сумма ===
    sum_patterns = [
        r'(?:ИТОГ|СУММА|СОММА|СОМСИ|СОМАНЫ|ХАЛИК|ОПЛАТА|Итог)[^\d]{0,5}=?\s*([0-9]+(?:[.,]\d{1,2})?)',
        r'=\s*([0-9]+(?:[.,]\d{1,2})?)'
    ]
    sums = []
    for pat in sum_patterns:
        for m in re.finditer(pat, clean, re.IGNORECASE):
            sums.append(float(m.group(1).replace(',', '.')))
    if sums:
        sums_sorted = sorted(sums, reverse=True)
        res["sum"] = f"{sums_sorted[0]:.2f}"
        res["fallback_sum"] = f"{sums_sorted[1]:.2f}" if len(sums_sorted) > 1 else f"{sums_sorted[0]:.2f}"

    return res

async def start_batch():
    IMAGES_FOLDER = "/checks"
    OUTPUT_JSON = "ocr_results.json"
    BATCH_SIZE = 3

    all_images = [
        os.path.join(IMAGES_FOLDER, f)
        for f in os.listdir(IMAGES_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # File doesn't exist, just ignore
        data = []

    exclude_image_names = [item["image"] for item in data if "image" in item]

    all_images = [
        path for path in all_images
        if os.path.basename(path) not in exclude_image_names
    ]

    random.shuffle(all_images)

    results = []
    for i in range(0, len(all_images), BATCH_SIZE):
        batch = all_images[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} images)...")
        batch_results = await process_batch(batch)
        results.extend(batch_results)

        # Save after each batch
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

async def process_batch(images: List[str]) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        tasks = [process_image(client, img) for img in images]
        return await asyncio.gather(*tasks)

def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0

async def process_image(client: httpx.AsyncClient, image_path: str) -> Dict:
    API_URL = "http://localhost:8000/api/ocr"
    filename = os.path.basename(image_path)

    try:
        # --- Open image and preprocess ---
        img = Image.open(image_path).convert("L")  # grayscale
        img_io = io.BytesIO()
        img.save(img_io, format="JPEG", optimize=True, quality=70)
        img_io.seek(0)

        # --- Prepare request ---
        files = {"image": (filename, img_io, "image/jpeg")}
        data = {
            "mode": "freeform",
            "prompt": "",
            "grounding": "false",
            "include_caption": "false",
            "find_term": "",
            "schema": "",
            "base_size": "1024",
            "image_size": "640",
            "crop_mode": "true",
            "test_compress": "false",
        }

        # --- Send request to local OCR ---
        response = await client.post(API_URL, data=data, files=files, timeout=60)
        response.raise_for_status()
        ocr_result = response.json()

        print("OCR:", ocr_result)  # helpful for debugging

        # The OCR service returns something like:
        # {'fb': '1112151856513', 'kkm': '11248', 'sum': '2175.00', 'fallback_sum': '2175.00', 'datetime': '02.09.2025 21:57'}

        fb = ocr_result.get("detected").get("fb")
        kkm = ocr_result.get("detected").get("kkm")
        total_sum = to_int(ocr_result.get("detected", {}).get("sum"))
        fallback_sum = to_int(ocr_result.get("detected", {}).get("fallback_sum"))
        datetime_str = ocr_result.get("detected").get("datetime")

        # Split datetime safely
        date_part, time_part = None, None
        if datetime_str and " " in datetime_str:
            date_part, time_part = datetime_str.split(" ", 1)

        # Prepare payload for remote API
        payload = {
            "type": "fields",
            "rnm": kkm,
            "fp": fb,
            "date": date_part,
            "time": time_part,
            "sum": total_sum,
            "fallback_sum": fallback_sum,
        }

        required_fields = [kkm, fb, date_part, time_part, total_sum]

        # Skip sending if any required field is empty
        if any(v in (None, "", []) for v in required_fields):
            print("⚠️ Skipping payload — missing required field(s):", payload)

            return {
                "status": "failed",
                "image": filename,
                "ocr_result": ocr_result["text"],
                "payload": ocr_result["detected"]
            }
        else:
            api_response = await send_payload(payload, url_template="https://oofd.underdev.kz/api/parser/fields")
            print("Parsed:", api_response)

            return {
                "image": filename,
                "ocr_result": ocr_result["text"],
                "payload": ocr_result["detected"],
                "parsed_response": api_response,
            }

    except Exception as e:
        return {"image": filename, "error": str(e)}

async def send_payload(payload: dict, url_template: str):
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {"Authorization": f"Bearer 4f6c18df0ebf9d6f2c4b7318c344c5d5f4b5a6d99c7e5c83a03a9cbd58e0d1"}
        print(payload)
        try:
            response = await client.post(url_template, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError):
            try:
                payload['sum'] = payload['fallback_sum']
                response = await client.post(url_template, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.RequestError):
                return {
                    "error": "Request failed 2 times"
                }

if __name__ == "__main__":
    host = env_config("API_HOST", default="0.0.0.0")
    port = env_config("API_PORT", default=8000, cast=int)
    uvicorn.run(app, host=host, port=port)
