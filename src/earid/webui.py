"""Web UI for reference-vs-gallery ear comparison.

Upload one reference photo and a batch of candidate photos; candidates are
ranked by cosine similarity between L2-normalized ear embeddings from a
trained checkpoint. Similarity scores are investigative-lead indicators, not
identity conclusions.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image, ImageOps
from torchvision import transforms

try:  # enable HEIC/HEIF uploads (iPhone photos) when pillow-heif is installed
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - optional dependency
    pass

from .models import build_model, extract_embeddings

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EarID Comparator</title>
<style>
  :root { --bg:#101418; --panel:#1a2027; --accent:#4da3ff; --text:#e8edf2; --muted:#8a97a5; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background:var(--bg); color:var(--text); min-height:100vh; }
  header { padding:14px 22px; border-bottom:1px solid #2a323c; display:flex; align-items:baseline; gap:14px; }
  header h1 { font-size:18px; margin:0; }
  header span { color:var(--muted); font-size:13px; }
  main { display:grid; grid-template-columns: 1fr 1.4fr; gap:18px; padding:18px 22px; }
  .panel { background:var(--panel); border:1px solid #2a323c; border-radius:12px; padding:16px; }
  .panel h2 { margin:0 0 10px; font-size:14px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
  .drop { border:2px dashed #34404d; border-radius:10px; min-height:120px; display:flex; flex-direction:column;
          align-items:center; justify-content:center; gap:6px; cursor:pointer; padding:14px; text-align:center;
          transition:border-color .15s; }
  .drop:hover, .drop.over { border-color:var(--accent); }
  .drop input { display:none; }
  .drop small { color:var(--muted); }
  #refPreview { max-width:100%; max-height:420px; border-radius:10px; margin-top:12px; display:none; }
  #rankBtn { margin-top:14px; width:100%; padding:12px; font-size:15px; font-weight:600; border:none;
             border-radius:10px; background:var(--accent); color:#04121f; cursor:pointer; }
  #rankBtn:disabled { background:#2c3743; color:var(--muted); cursor:not-allowed; }
  #status { margin-top:10px; font-size:13px; color:var(--muted); min-height:18px; }
  .viewer { display:none; flex-direction:column; align-items:center; gap:12px; }
  .stage { position:relative; width:100%; display:flex; align-items:center; justify-content:center; }
  .stage img { max-width:100%; max-height:480px; border-radius:10px; }
  .navbtn { position:absolute; top:50%; transform:translateY(-50%); background:#0a0e12cc; color:var(--text);
            border:1px solid #34404d; width:44px; height:64px; border-radius:10px; font-size:22px; cursor:pointer; }
  .navbtn:hover { border-color:var(--accent); }
  #prevBtn { left:8px; } #nextBtn { right:8px; }
  .meta { display:flex; gap:18px; align-items:center; flex-wrap:wrap; justify-content:center; }
  .badge { background:#0a0e12; border:1px solid #34404d; border-radius:8px; padding:6px 12px; font-size:13px; }
  .badge b { color:var(--accent); }
  .scorebar { width:100%; max-width:460px; height:8px; background:#0a0e12; border-radius:4px; overflow:hidden; }
  .scorebar div { height:100%; background:linear-gradient(90deg,#2f6fb3,var(--accent)); }
  #thumbs { display:flex; gap:6px; overflow-x:auto; width:100%; padding:4px 0; }
  #thumbs img { height:56px; border-radius:6px; opacity:.45; cursor:pointer; border:2px solid transparent; }
  #thumbs img.active { opacity:1; border-color:var(--accent); }
  footer { padding:10px 22px 20px; color:var(--muted); font-size:12px; }
  @media (max-width: 900px) { main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header><h1>EarID Comparator</h1><span>{{ backbone }} &middot; {{ run_name }}</span></header>
<main>
  <section class="panel">
    <h2>Reference photo</h2>
    <label class="drop" id="refDrop">
      <input type="file" id="refInput" accept="image/*,.heic,.heif">
      <div>Click or drop the reference ear photo</div>
      <small id="refName">No file selected</small>
    </label>
    <img id="refPreview" alt="reference preview">
    <button id="rankBtn" disabled>Rank candidates</button>
    <div id="status"></div>
  </section>
  <section class="panel">
    <h2>Candidate gallery</h2>
    <label class="drop" id="candDrop">
      <input type="file" id="candInput" accept="image/*,.heic,.heif" multiple>
      <div>Click or drop hundreds of candidate photos</div>
      <small id="candCount">No files selected</small>
    </label>
    <div class="viewer" id="viewer">
      <div class="meta">
        <span class="badge">Rank <b id="rankPos">1</b> / <span id="rankTotal">0</span></span>
        <span class="badge">Similarity <b id="scoreVal">–</b></span>
        <span class="badge" id="fileName"></span>
      </div>
      <div class="scorebar"><div id="scoreFill" style="width:0%"></div></div>
      <div class="stage">
        <button class="navbtn" id="prevBtn">&#8592;</button>
        <img id="candImage" alt="candidate">
        <button class="navbtn" id="nextBtn">&#8594;</button>
      </div>
      <div id="thumbs"></div>
    </div>
  </section>
</main>
<footer>Similarity scores are investigative-lead indicators only; they are not identity conclusions
and must not be treated as forensic identifications.</footer>
<script>
const refInput = document.getElementById('refInput');
const candInput = document.getElementById('candInput');
const rankBtn = document.getElementById('rankBtn');
const statusEl = document.getElementById('status');
let refFile = null, candFiles = [], ranked = [], urls = [], pos = 0;
const HEIC = /\\.(heic|heif)$/i;
const isImage = f => f.type.startsWith('image/') || HEIC.test(f.name);

// Browsers cannot decode HEIC/HEIF; ask the server for a JPEG preview instead.
async function previewURL(file) {
  const needsConvert = HEIC.test(file.name) || file.type === 'image/heic' || file.type === 'image/heif';
  if (!needsConvert) return URL.createObjectURL(file);
  const form = new FormData();
  form.append('image', file);
  const res = await fetch('/api/preview', { method: 'POST', body: form });
  if (!res.ok) return URL.createObjectURL(file);
  return URL.createObjectURL(await res.blob());
}

function hook(dropId, inputEl, onFiles) {
  const drop = document.getElementById(dropId);
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('over');
    onFiles([...e.dataTransfer.files].filter(isImage));
  });
  inputEl.addEventListener('change', () => onFiles([...inputEl.files].filter(isImage)));
}

hook('refDrop', refInput, async files => {
  if (!files.length) return;
  refFile = files[0];
  document.getElementById('refName').textContent = refFile.name;
  const img = document.getElementById('refPreview');
  img.style.display = 'block';
  img.src = await previewURL(refFile);
  updateButton();
});

hook('candDrop', candInput, files => {
  if (!files.length) return;
  candFiles = files;
  document.getElementById('candCount').textContent = files.length + ' file(s) selected';
  updateButton();
});

function updateButton() { rankBtn.disabled = !(refFile && candFiles.length); }

rankBtn.addEventListener('click', async () => {
  rankBtn.disabled = true;
  statusEl.textContent = 'Scoring ' + candFiles.length + ' candidates…';
  const form = new FormData();
  form.append('reference', refFile);
  candFiles.forEach(f => form.append('candidates', f));
  try {
    const res = await fetch('/api/rank', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const data = await res.json();
    ranked = data.ranking;
    statusEl.textContent = 'Preparing previews…';
    urls.forEach(u => URL.revokeObjectURL(u));
    urls = await Promise.all(candFiles.map(previewURL));
    pos = 0;
    buildThumbs();
    show();
    document.getElementById('viewer').style.display = 'flex';
    statusEl.textContent = 'Ranked ' + ranked.length + ' candidates in ' + data.seconds.toFixed(1) + 's';
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  } finally { rankBtn.disabled = false; }
});

function buildThumbs() {
  const wrap = document.getElementById('thumbs');
  wrap.innerHTML = '';
  ranked.forEach((r, i) => {
    const t = document.createElement('img');
    t.src = urls[r.index];
    t.title = '#' + (i + 1) + ' ' + r.score.toFixed(4);
    t.addEventListener('click', () => { pos = i; show(); });
    wrap.appendChild(t);
  });
}

function show() {
  const r = ranked[pos];
  document.getElementById('candImage').src = urls[r.index];
  document.getElementById('rankPos').textContent = pos + 1;
  document.getElementById('rankTotal').textContent = ranked.length;
  document.getElementById('scoreVal').textContent = r.score.toFixed(4);
  document.getElementById('fileName').textContent = r.name;
  document.getElementById('scoreFill').style.width = Math.max(0, Math.min(1, (r.score + 1) / 2)) * 100 + '%';
  [...document.getElementById('thumbs').children].forEach((t, i) => {
    t.classList.toggle('active', i === pos);
    if (i === pos) t.scrollIntoView({ inline: 'center', behavior: 'smooth', block: 'nearest' });
  });
}

function step(d) { if (!ranked.length) return; pos = (pos + d + ranked.length) % ranked.length; show(); }
document.getElementById('prevBtn').addEventListener('click', () => step(-1));
document.getElementById('nextBtn').addEventListener('click', () => step(1));
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
});
let touchX = null;
document.getElementById('candImage').addEventListener('touchstart', e => touchX = e.touches[0].clientX);
document.getElementById('candImage').addEventListener('touchend', e => {
  if (touchX === null) return;
  const dx = e.changedTouches[0].clientX - touchX;
  if (Math.abs(dx) > 40) step(dx < 0 ? 1 : -1);
  touchX = null;
});
</script>
</body>
</html>"""


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = checkpoint["config"]["backbone"]
    model = build_model(
        backbone,
        num_classes=len(checkpoint["label_to_index"]),
        pretrained=False,
        loss=checkpoint["config"].get("loss", "ce"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    image_size = int(checkpoint["config"].get("image_size", 224))
    return model, backbone, image_size


def _build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def create_app(
    checkpoint_path: Path,
    device_name: str = "cpu",
    batch_size: int = 16,
    align_checkpoint: Path | None = None,
) -> Flask:
    device = torch.device(device_name)
    model, backbone, image_size = _load_checkpoint(checkpoint_path, device)
    transform = _build_transform(image_size)
    run_name = checkpoint_path.parent.name

    aligner = None
    if align_checkpoint is not None:
        from .align import align_image, load_landmark_model

        landmark_model, landmark_size = load_landmark_model(align_checkpoint, device)

        def aligner(image: Image.Image) -> Image.Image:
            return align_image(landmark_model, image, landmark_size, device, output_size=image_size)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

    def embed(tensors: list[torch.Tensor]) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start : start + batch_size]).to(device)
                features = extract_embeddings(model, backbone, batch)
                chunks.append(torch.nn.functional.normalize(features.cpu(), dim=1))
        return torch.cat(chunks, dim=0)

    @app.get("/")
    def index():
        return render_template_string(PAGE, backbone=backbone, run_name=run_name)

    @app.post("/api/rank")
    def rank():
        import time

        started = time.time()
        reference = request.files.get("reference")
        candidates = request.files.getlist("candidates")
        if reference is None or not candidates:
            return jsonify({"error": "Provide one reference and at least one candidate image"}), 400

        def to_tensor(storage) -> torch.Tensor:
            with Image.open(io.BytesIO(storage.read())) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                if aligner is not None:
                    image = aligner(image)
                return transform(image)

        try:
            reference_tensor = to_tensor(reference)
            candidate_tensors = [to_tensor(f) for f in candidates]
        except Exception as error:  # noqa: BLE001 - report unreadable uploads to the client
            return jsonify({"error": f"Unreadable image: {error}"}), 400

        embeddings = embed([reference_tensor] + candidate_tensors)
        similarities = (embeddings[1:] @ embeddings[0]).tolist()
        order = sorted(range(len(candidates)), key=lambda i: similarities[i], reverse=True)
        ranking = [
            {"index": i, "name": candidates[i].filename, "score": similarities[i]} for i in order
        ]
        return jsonify({"ranking": ranking, "seconds": time.time() - started})

    @app.post("/api/preview")
    def preview():
        upload = request.files.get("image")
        if upload is None:
            return jsonify({"error": "Provide an image"}), 400
        try:
            # Previews always show the user's original photo; alignment is
            # applied only in the backend embedding path.
            with Image.open(io.BytesIO(upload.read())) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1280, 1280))
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=88)
        except Exception as error:  # noqa: BLE001 - report unreadable uploads to the client
            return jsonify({"error": f"Unreadable image: {error}"}), 400
        return Response(buffer.getvalue(), mimetype="image/jpeg")

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EarID reference-vs-gallery comparison UI")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--align-checkpoint", help="Optional landmarks.pt for automatic ear alignment of uploads")
    args = parser.parse_args(argv)

    app = create_app(
        Path(args.checkpoint),
        args.device,
        args.batch_size,
        align_checkpoint=Path(args.align_checkpoint) if args.align_checkpoint else None,
    )
    print(json.dumps({"url": f"http://{args.host}:{args.port}", "checkpoint": args.checkpoint}))
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
