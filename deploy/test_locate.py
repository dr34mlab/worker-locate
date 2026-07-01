#!/usr/bin/env python3
"""Fire a test grounding request at the locate-anything endpoint.

Usage:
  python3 deploy/test_locate.py [--image path/or/url] [--queries term1,term2]
  python3 deploy/test_locate.py --image http://10.1.1.99:8900/frame.jpg

Requires earl key: RUNPOD_API_KEY_CLI, runpod_locate_endpoint_id
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
import urllib.request


def earl(account: str) -> str:
    return subprocess.check_output(
        ["security", "find-generic-password", "-s", "earl", "-a", account, "-w"],
        timeout=5,
    ).decode().strip()


DEFAULT_QUERIES = [
    "clothes", "dishes", "cups", "bottles", "cans", "boxes",
    "bags", "trash", "shoes", "papers", "plates", "blankets",
]

DEFAULT_IMAGE_URL = "http://10.1.1.99:8900/frame.jpg"


def fetch_image_b64(source: str) -> str:
    """Fetch image from path or URL, return as base64 JPEG."""
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=10) as r:
            raw = r.read()
    else:
        with open(source, "rb") as f:
            raw = f.read()

    # Re-encode as JPEG to normalize format
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def run_sync(ep_id: str, key: str, payload: dict) -> dict:
    """POST to /runsync and return output."""
    url = f"https://api.runpod.ai/v2/{ep_id}/runsync"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    elapsed = time.time() - t0
    return resp, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE_URL)
    ap.add_argument("--queries", default=",".join(DEFAULT_QUERIES))
    ap.add_argument("--endpoint", default="", help="Override endpoint id (else reads earl)")
    args = ap.parse_args()

    key = earl("RUNPOD_API_KEY_CLI")
    ep_id = args.endpoint or earl("runpod_locate_endpoint_id")
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    print(f"Endpoint: {ep_id}")
    print(f"Image:    {args.image}")
    print(f"Queries:  {queries}")
    print()

    print("Fetching image ...")
    try:
        b64 = fetch_image_b64(args.image)
        print(f"Image fetched ({len(b64)//1024} KB base64)")
    except Exception as e:
        print(f"ERROR fetching image: {e}", file=sys.stderr)
        sys.exit(1)

    print("Sending to endpoint ...")
    resp, elapsed = run_sync(ep_id, key, {"input": {"image_b64": b64, "queries": queries}})

    print(f"Response ({elapsed:.1f}s):")
    print(json.dumps(resp, indent=2))

    if resp.get("status") == "COMPLETED":
        out = resp.get("output", {})
        print()
        print(f"Detections: {out.get('count', '?')}")
        for d in out.get("detections", []):
            b = d["box"]
            print(f"  {d['label']}: [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]")


if __name__ == "__main__":
    main()
