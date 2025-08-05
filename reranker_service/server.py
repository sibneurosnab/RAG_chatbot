import os, gc, logging, torch, torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_client import Counter, start_http_server

MODEL_NAME  = "jinaai/jina-reranker-v2-base-multilingual"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float16 if DEVICE == "cuda" else torch.float32
RATE_LIMIT  = "60/minute"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("JinaReranker")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    trust_remote_code=True
).to(DEVICE).eval()
log.info("Model %s loaded (%s, %s)", MODEL_NAME, DEVICE, DTYPE)

app = Flask(__name__)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT],
)
limiter.init_app(app)

REQ_CNT = Counter("rerank_requests_total", "Total rerank requests")

@torch.inference_mode()
def rerank_pairs(query, documents):
    scores = []
    for doc in documents:
        encoded = tokenizer(query, doc, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        logits = model(**encoded).logits
        prob = logits[0].item()
        scores.append(prob)
    return scores

@app.route("/rerank", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def rerank():
    REQ_CNT.inc()
    data = request.get_json(force=True)
    scores = rerank_pairs(data["query"], data["documents"])
    gc.collect()
    return jsonify({"results": [{"document": doc, "score": s} for doc, s in zip(data["documents"], scores)]})

if __name__ == "__main__":
    start_http_server(8009)
    app.run(host="0.0.0.0", port=8010)
