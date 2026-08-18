"""The WHOLE WHOLE journey, now with eyes.

A tiny zero-dependency web app that mirrors step07c_the_whole_whole_journey.py
visually, one step at a time:

    text -> token IDs -> sliding windows -> a batch -> token embeddings
    -> + positional embeddings -> [batch_size, max_length, 256]

Run me:
    python app.py
then open http://127.0.0.1:8765

The backend reuses the exact logic of the reference script (same seed, same
tokenizer, same shapes) and serves every intermediate as JSON. The frontend
(static/index.html) walks through the steps with animations and heatmaps.
"""
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tiktoken
import torch

BASE = Path(__file__).resolve().parent
TEXT_FILE = BASE / "the-verdict.txt"
INDEX_FILE = BASE / "static" / "index.html"
PORT = 8765

# --- same seeds / tables as the reference script, in the same order ----------
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

VOCAB_SIZE = 50257       # GPT-2's dictionary
DEFAULT_OUTPUT_DIM = 256 # numbers describing one token
DISPLAY_DIM = 64         # heatmaps show 64 of the 256 numbers (chunk-averaged)


def downsample(vec, display_dim=DISPLAY_DIM):
    """Average chunks of the 256-dim vectors down to display_dim columns."""
    v = torch.as_tensor(vec, dtype=torch.float32)
    n = v.shape[-1]
    k = n // display_dim
    return v.view(*v.shape[:-1], display_dim, k).mean(dim=-1)


def r3(x):
    return round(float(x), 3)


def build_journey(batch_size=8, max_length=4):
    """Recreate every step of the reference script and return it as JSON."""
    batch_size = max(1, min(int(batch_size), 64))
    max_length = max(2, min(int(max_length), 16))
    stride = max_length

    # STEP 1 ------------------------------------------------------------------
    raw_text = TEXT_FILE.read_text(encoding="utf-8")

    # STEP 2 ------------------------------------------------------------------
    token_ids = tokenizer.encode(raw_text)
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]

    # STEP 3 ------------------------------------------------------------------
    inputs_pile, targets_pile = [], []
    for i in range(0, len(token_ids) - max_length, stride):
        inputs_pile.append(token_ids[i:i + max_length])
        targets_pile.append(token_ids[i + 1:i + max_length + 1])

    # STEP 4 ------------------------------------------------------------------
    batch_size = min(batch_size, len(inputs_pile))
    total_batches = len(inputs_pile) // batch_size

    # STEP 5 ------------------------------------------------------------------
    inputs = torch.tensor(inputs_pile[:batch_size])     # [B, L] ids
    targets = torch.tensor(targets_pile[:batch_size])   # [B, L] ids

    def decode_row(ids):
        return [tokenizer.decode([int(i)]) for i in ids]

    row0_in = inputs[0].tolist()
    row0_questions = [
        {"ctx": tokenizer.decode(row0_in[:pos + 1]),
         "ans": tokenizer.decode([int(targets[0, pos])])}
        for pos in range(max_length)
    ]

    # STEP 6 ------------------------------------------------------------------
    token_emb_layer = torch.nn.Embedding(VOCAB_SIZE, DEFAULT_OUTPUT_DIM)
    token_embeddings = token_emb_layer(inputs)          # [B, L, 256]

    # STEP 7 ------------------------------------------------------------------
    pos_emb_layer = torch.nn.Embedding(max_length, DEFAULT_OUTPUT_DIM)
    pos_embeddings = pos_emb_layer(torch.arange(max_length))  # [L, 256]
    input_embeddings = token_embeddings + pos_embeddings      # [B, L, 256]

    # PROOF -------------------------------------------------------------------
    same = torch.tensor([[588] * max_length])            # " like" x L
    tok_proof = token_emb_layer(same)                    # [1, L, 256]
    fixed_proof = tok_proof + pos_embeddings             # [1, L, 256]

    return {
        "meta": {
            "char_count": len(raw_text),
            "token_count": len(token_ids),
            "first_ids": token_ids[:8],
            "windows": len(inputs_pile),
            "batch_size": batch_size,
            "max_length": max_length,
            "stride": stride,
            "total_batches": total_batches,
            "vocab_size": VOCAB_SIZE,
            "output_dim": DEFAULT_OUTPUT_DIM,
            "display_dim": DISPLAY_DIM,
        },
        "text": {"preview": raw_text[:1200]},
        "tokens": [
            {"id": tid, "text": txt}
            for tid, txt in zip(token_ids[:220], token_texts[:220])
        ],
        "windows": [
            {
                "input_ids": inputs_pile[w],
                "target_ids": targets_pile[w],
                "input_texts": decode_row(inputs_pile[w]),
                "target_texts": decode_row(targets_pile[w]),
            }
            for w in range(min(10, len(inputs_pile)))
        ],
        "batch": {
            "inputs": [r.tolist() for r in inputs],
            "targets": [r.tolist() for r in targets],
            "decoded": [decode_row(r) for r in inputs],
            "row0_questions": row0_questions,
        },
        "embeddings": {
            "token_emb": downsample(token_embeddings).tolist(),   # [B, L, 64]
            "pos_emb": downsample(pos_embeddings).tolist(),       # [L, 64]
            "input_emb": downsample(input_embeddings).tolist(),   # [B, L, 64]
            "token_full_row0": [[r3(v) for v in token_embeddings[0, 0, :8].tolist()]],
            "input_full_row0": [[r3(v) for v in input_embeddings[0, 0, :8].tolist()]],
        },
        "proof": {
            "token_first3": [[r3(v) for v in tok_proof[0, s, :3].tolist()]
                             for s in range(max_length)],
            "input_first3": [[r3(v) for v in fixed_proof[0, s, :3].tolist()]
                             for s in range(max_length)],
            "token_display": downsample(tok_proof[0]).tolist(),   # [L, 64]
            "input_display": downsample(fixed_proof[0]).tolist(), # [L, 64]
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        print("[http]", fmt % args)

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_FILE.read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
        elif parsed.path == "/api/journey":
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                data = build_journey(
                    batch_size=int(qs.get("batch_size", ["8"])[0]),
                    max_length=int(qs.get("max_length", ["4"])[0]),
                )
                body = json.dumps(data).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001 - surface to the frontend
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(500, body, "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    if not TEXT_FILE.exists():
        raise SystemExit(f"missing {TEXT_FILE} - copy it next to this file")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("=" * 64)
    print("THE WHOLE WHOLE JOURNEY - visual walkthrough")
    print(f"  open  http://127.0.0.1:{PORT}")
    print("  press Ctrl+C to stop")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
