#!/usr/bin/env python3
"""Generate a series of Higgsfield clips from a scenes file.

Usage:
    python generate_series.py scenes.json [--out out] [--workers 3] [--dry-run]

Credentials: HF_KEY="key:secret"  or  HF_API_KEY + HF_API_SECRET.

Scenes file (see scenes.example.json):
  {
    "model": "<higgsfield model id>",
    "defaults": {...model arguments applied to every scene...},
    "style": "text appended to every prompt (look, lighting, lens)",
    "characters": {"anna": {"description": "...", "reference": "refs/anna.png"}},
    "scenes": [{"id": "s01", "prompt": "...", "characters": ["anna"], "arguments": {...}}]
  }

Consistency comes from: a shared style suffix, character descriptions injected
into each prompt, a fixed seed, and reference images uploaded once and passed
to the model through `reference_argument` (name depends on the model).
Finished scenes are skipped on re-run (resume), results are logged to manifest.json.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import httpx


def find_media_urls(obj):
    """Collect every http(s) URL found under a result dict (video first)."""
    urls = []
    if isinstance(obj, dict):
        for v in obj.values():
            urls += find_media_urls(v)
    elif isinstance(obj, list):
        for v in obj:
            urls += find_media_urls(v)
    elif isinstance(obj, str) and obj.startswith(("http://", "https://")):
        urls.append(obj)
    return urls


def build_prompt(scene, cfg):
    parts = [scene["prompt"]]
    for name in scene.get("characters", []):
        desc = cfg.get("characters", {}).get(name, {}).get("description")
        if desc:
            parts.append(f"{name}: {desc}")
    if cfg.get("style"):
        parts.append(cfg["style"])
    return ". ".join(p.rstrip(". ") for p in parts)


def upload_references(cfg, hf):
    """Upload each character reference image once; returns {name: url}."""
    urls = {}
    for name, ch in cfg.get("characters", {}).items():
        ref = ch.get("reference")
        if not ref:
            continue
        if ref.startswith("http"):
            urls[name] = ref
        else:
            print(f"uploading reference for {name}: {ref}")
            urls[name] = hf.upload_file(ref)
    return urls


def run_scene(scene, cfg, ref_urls, out_dir, hf, dry_run):
    sid = scene["id"]
    dest = out_dir / f"{sid}.mp4"
    args = dict(cfg.get("defaults", {}))
    if "seed" in cfg:
        args.setdefault("seed", cfg["seed"])
    args.update(scene.get("arguments", {}))
    args["prompt"] = build_prompt(scene, cfg)

    ref_arg = cfg.get("reference_argument")  # e.g. "image_url"
    refs = [ref_urls[c] for c in scene.get("characters", []) if c in ref_urls]
    if ref_arg and refs and ref_arg not in args:
        args[ref_arg] = refs[0]

    if dry_run:
        return {"id": sid, "status": "dry-run", "model": cfg["model"], "arguments": args}
    if dest.exists():
        return {"id": sid, "status": "skipped", "file": str(dest)}

    t0 = time.time()
    result = hf.subscribe(
        cfg["model"],
        arguments=args,
        on_enqueue=lambda rid: print(f"[{sid}] enqueued {rid}"),
        on_queue_update=lambda s: print(f"[{sid}] {type(s).__name__}"),
    )
    urls = find_media_urls(result)
    video = next((u for u in urls if urlparse(u).path.lower().endswith((".mp4", ".mov", ".webm"))), None)
    video = video or (urls[0] if urls else None)
    if not video:
        raise RuntimeError(f"no media URL in result: {result}")
    with httpx.stream("GET", video, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return {"id": sid, "status": "done", "file": str(dest), "url": video,
            "seconds": round(time.time() - t0), "arguments": args}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes")
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="print requests, spend no credits")
    ns = ap.parse_args()

    cfg = json.loads(Path(ns.scenes).read_text())
    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf = None
    ref_urls = {}
    if not ns.dry_run:
        import higgsfield_client as hf  # noqa: F811 (imported lazily so --dry-run needs no SDK)
        ref_urls = upload_references(cfg, hf)

    results = []
    with ThreadPoolExecutor(max_workers=ns.workers) as pool:
        futs = {pool.submit(run_scene, s, cfg, ref_urls, out_dir, hf, ns.dry_run): s for s in cfg["scenes"]}
        for fut in as_completed(futs):
            sid = futs[fut]["id"]
            try:
                res = fut.result()
            except Exception as e:  # keep going; failed scenes are retried on next run
                res = {"id": sid, "status": "failed", "error": str(e)}
            print(f"[{sid}] {res['status']}")
            results.append(res)

    results.sort(key=lambda r: r["id"])
    (out_dir / "manifest.json").write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r["status"] == "failed"]
    print(f"{len(results) - len(failed)}/{len(results)} ok; manifest at {out_dir / 'manifest.json'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
