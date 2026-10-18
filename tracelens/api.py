"""Local-only API. Saved evaluation artifacts and live checkpoint inference share a scorer."""
from __future__ import annotations

import json
import gzip
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import normalize, sha256
from .models import SequenceModel
from .train import score_model
from .calibration import aggregate
from .semantic import TextCNN,encode_groups,score_text, WORDS, TOKENS_PER_EVENT


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500_000)
    model: str


def create_app(artifacts: Path) -> FastAPI:
    artifacts = artifacts.resolve()
    required = ["report.json", "vocab.json", "calibration.json", "baseline.json"]
    if not all((artifacts / name).is_file() for name in required):
        raise FileNotFoundError("Evaluation artifacts missing. Run tracelens train first, or use artifacts/portfolio.")
    manifest_path = artifacts / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            if sha256(artifacts / name) != expected:
                raise ValueError(f'Artifact checksum mismatch: {name}')
    report = json.loads((artifacts / "report.json").read_text(encoding="utf-8"))
    if (artifacts/'traces.json.gz').exists():
        with gzip.open(artifacts/'traces.json.gz','rt',encoding='utf-8') as f:traces=json.load(f)
    else:traces = json.loads((artifacts / "traces.json").read_text(encoding="utf-8"))
    vocab = json.loads((artifacts / "vocab.json").read_text(encoding="utf-8"))
    catalog=None
    if isinstance(traces,dict) and traces.get('format')=='compact-v2':
        payload=traces;catalog=payload['templates']
        traces=[{'id':i,'start_line':row[1],'end_line':row[1]+report['dataset']['window_size']-1,'label':row[2],
                 'split':payload['splits'][row[0]],'event_ids':row[3],'label_mask':row[4],
                 'scores':dict(zip(payload['models'],row[5]))} for i,row in enumerate(payload['windows'])]
    def events_for(trace):
        if 'events' in trace:return trace['events']
        return [{'line':trace['start_line']+j,'template':catalog[idx],
                 'token':vocab.get(catalog[idx],0),'known':bool(vocab.get(catalog[idx],0)),
                 'label':int(bool(trace['label_mask']&(1<<j)))} for j,idx in enumerate(trace['event_ids'])]
    def templates_for(trace):
        return [catalog[i] for i in trace['event_ids']] if catalog is not None else [e['template'] for e in trace['events']]
    word_vocab = json.loads((artifacts / 'word_vocab.json').read_text(encoding='utf-8')) if (artifacts/'word_vocab.json').exists() else None
    calibration = json.loads((artifacts / "calibration.json").read_text(encoding="utf-8"))
    probability = np.array(json.loads((artifacts / "baseline.json").read_text(encoding="utf-8"))["probability"])
    runs = {run["id"]: run for run in report["runs"]}
    torch.set_num_threads(4)
    app = FastAPI(title="TraceLens", version="0.2.1", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        # Bound requests before parsing JSON; run on loopback by default.
        length = request.headers.get("content-length", "0")
        if not length.isdigit() or int(length) > 1_000_000:
            return JSONResponse({"detail": "Request exceeds the 1 MB limit."}, status_code=413)
        if request.method in {'POST', 'PUT', 'PATCH'}:
            chunks = []
            received = 0
            async for chunk in request.stream():
                received += len(chunk)
                if received > 1_000_000:
                    return JSONResponse({'detail': 'Request exceeds the 1 MB limit.'}, status_code=413)
                chunks.append(chunk)
            request._body = b''.join(chunks)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        return response

    def get_run(model):
        if model not in runs:
            raise HTTPException(404, "Unknown model checkpoint.")
        return runs[model]

    @lru_cache(maxsize=12)
    def load_checkpoint(filename):
        payload = torch.load(artifacts / filename, map_location="cpu", weights_only=True)
        net = TextCNN(payload['vocab_size']) if payload.get('model_type')=='semantic' else SequenceModel(payload["vocab_size"], **payload["model_config"])
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return net

    def infer(model, x, groups=None):
        entry=get_run(model)
        if model == "unigram":
            losses = -np.log(probability[x[:, 1:]])
            return losses.mean(axis=1), losses
        if entry.get('model_type')=='semantic':
            encoded=encode_groups(groups,word_vocab)
            events=np.mean([score_text(load_checkpoint(name),encoded)[1] for name in entry['checkpoints']],axis=0)
            return aggregate(events,entry['aggregation']),events
        if entry.get('checkpoints'):
            events=np.mean([score_model(load_checkpoint(name),x)[1] for name in entry['checkpoints']],axis=0)
            return aggregate(events,entry.get('aggregation','mean')),events
        return score_model(load_checkpoint(entry['checkpoint']), x)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models": len(runs), "artifact_schema": report["schema_version"]}

    @app.get("/api/docs", response_class=HTMLResponse, include_in_schema=False)
    def docs():
        content = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TraceLens API</title><style>body{font:16px/1.8 system-ui;max-width:850px;margin:60px auto;padding:24px;color:#183437;background:#f5f6f2}code,pre{background:#e8eee3;padding:4px 8px;border-radius:4px}pre{overflow:auto;padding:20px}li{margin:16px 0}a{color:#157b62}</style></head><body><a href="/">← TraceLens workspace</a><h1>TraceLens API</h1><p>A local API for saved research artifacts and real checkpoint inference.</p><ul><li><code>GET /api/health</code> — availability and model count</li><li><code>GET /api/report</code> — complete experiment report</li><li><code>GET /api/timeline?model=lstm-s17</code> — held-out scores and normal-validation calibration values</li><li><code>GET /api/traces?model=lstm-s17&amp;kind=alerts</code> — paginated/searchable test windows</li><li><code>GET /api/traces/583?model=lstm-s17</code> — event content and checkpoint-based per-event surprise</li><li><code>POST /api/analyze</code> — JSON body with model ID and log text (at most 2,000 nonempty lines)</li></ul><pre>{"model": "lstm-s17", "text": "one log message per line\\n..."}</pre><p>The request needs at least one full context window. Scores are not failure probabilities.</p><p><a href="/openapi.json">Machine-readable OpenAPI schema →</a></p></body></html>'''
        return content.replace('lstm-s17', report['selected_model'])

    @app.get("/api/report")
    def summary():
        return report

    @app.get("/api/timeline")
    def timeline(model: str):
        entry = get_run(model)
        return {"threshold": entry["metrics"]["threshold"], "normal_calibration": calibration[model],
                "points": [{"id": t["id"], "score": t["scores"][model], "label": t["label"],
                            "line": t["start_line"],"split":t.get('split','test')} for t in traces]}

    @app.get("/api/traces")
    def listing(model: str, kind: str = "all", q: str = "", offset: int = Query(0, ge=0),
                limit: int = Query(30, ge=1, le=100), threshold: float | None = Query(None, ge=0),split: str = 'all'):
        entry = get_run(model)
        if kind not in {'all', 'alerts', 'missed', 'false'}:
            raise HTTPException(422, 'Unknown window filter.')
        if split not in {'all', *(t.get('split', 'test') for t in traces)}:
            raise HTTPException(422, 'Unknown audit range.')
        if report['schema_version'] == 2 and threshold is not None:
            raise HTTPException(422, 'Audit thresholds are frozen; overrides are unavailable.')
        cutoff = entry["metrics"]["threshold"] if threshold is None else threshold
        def matches(t):
            if split!='all' and t.get('split','test')!=split:return False
            flag = t["scores"][model] > cutoff
            if kind == "alerts" and not flag:
                return False
            if kind == "missed" and (not t["label"] or flag):
                return False
            if kind == "false" and (t["label"] or not flag):
                return False
            return not q or q.lower() in " ".join(templates_for(t)).lower() or q in str(t["start_line"])
        found = [t for t in traces if matches(t)]
        items = [{"id": t["id"], "start_line": t["start_line"], "end_line": t["end_line"], "time": t.get("time",""),
                  "label": t["label"], "score": t["scores"][model], "unknowns": sum(not e["known"] for e in events_for(t)),"split":t.get('split','test'),
                  "preview": events_for(t)[0]["template"]}
                 for t in found[offset:offset + limit]]
        return {"items": items, "total": len(found), "offset": offset}

    @app.get("/api/traces/{trace_id}")
    def detail(trace_id: int, model: str):
        if not 0 <= trace_id < len(traces):
            raise HTTPException(404, "Window not found.")
        trace = traces[trace_id]
        events=events_for(trace)
        x = np.array([[e["token"] for e in events]], dtype=np.int64)
        _, losses = infer(model, x, [events])
        semantic=get_run(model).get('model_type')=='semantic'
        return {**trace, 'events':events,"event_scores": losses[0].tolist() if semantic else [None] + losses[0].tolist(),
                'score_kind':'event activation' if semantic else 'next-event surprise'}

    @app.post("/api/analyze")
    def analyze(payload: AnalyzeRequest):
        entry = get_run(payload.model)
        source_lines = [(i, line.strip()) for i, line in enumerate(payload.text.splitlines(), 1) if line.strip()]
        raw_lines = [line for _, line in source_lines]
        size = report["dataset"]["window_size"]
        if len(raw_lines) < size:
            raise HTTPException(422, f"Provide at least {size} nonempty log lines.")
        if len(raw_lines) > 2000:
            raise HTTPException(422, "Analyze at most 2,000 lines at a time.")
        templates = []
        for line in raw_lines:
            fields = line.split(maxsplit=9)
            content = (fields[9] if len(fields)==10 else '<empty>') if len(fields)>=9 and fields[1].isdigit() else line
            templates.append(normalize(content))
        tokens = [vocab.get(t, 0) for t in templates]
        n = len(tokens) // size * size
        x = np.array(tokens[:n], dtype=np.int64).reshape(-1, size)
        groups=[[{'template':t} for t in templates[start:start+size]] for start in range(0,n,size)]
        scores, losses = infer(payload.model, x, groups)
        semantic=entry.get('model_type')=='semantic'
        # Coverage is a descriptive check, not an estimated probability of validity.
        # Entirely unseen/unencodable windows have no supported decision.
        coverage = []
        for group_index, group in enumerate(groups):
            full = [WORDS.findall(e['template'].lower()) for e in group]
            retained = [w for tokens in full for w in tokens[:TOKENS_PER_EVENT]]
            known_words = sum(word_vocab.get(w, 1) > 1 for w in retained) if semantic else None
            supported = (bool(retained) and known_words > 0 and all(full)) if semantic else bool((x[group_index] != 0).any())
            coverage.append({
                'status': 'unscorable' if not supported else 'domain_unverified',
                'known_word_fraction': known_words / len(retained) if semantic and retained else None,
                'empty_token_events': sum(not tokens for tokens in full),
                'truncated_events': sum(len(tokens) > TOKENS_PER_EVENT for tokens in full) if semantic else 0,
            })
        results = [{"id": i, "score": float(s), "alert": bool(s > entry["metrics"]["threshold"]) if coverage[i]['status'] != 'unscorable' else None,
                    "coverage": coverage[i],
                    "events": [{"line": source_lines[i * size + j][0], "template": templates[i * size + j],
                                "known": tokens[i * size + j] != 0,
                                "surprise": float(losses[i,j]) if semantic else (None if j == 0 else float(losses[i, j - 1]))} for j in range(size)]}
                   for i, s in enumerate(scores)]
        return {"model": payload.model, "threshold": entry["metrics"]["threshold"], "windows": results,
                "analyzed_lines": n, "discarded_tail_lines": len(tokens) - n,
                "unknown_fraction": float((x == 0).mean()),
                "legacy_template_unknown_fraction": float((x == 0).mean()),
                "ignored_blank_lines": len(payload.text.splitlines()) - len(raw_lines),
                "note": "Research replay only: the domain is unverified. Word coverage does not establish in-domain validity. Raw scores remain visible for diagnostics; unscorable windows receive no alert decision. Unknown-template fraction refers only to the legacy event vocabulary. Ground truth is unavailable."}

    web = Path(__file__).parent / "web"
    @app.get("/")
    def index():
        return FileResponse(web / ("v2.html" if report['schema_version']==2 else 'index.html'))

    app.mount("/static", StaticFiles(directory=web), name="static")
    return app
