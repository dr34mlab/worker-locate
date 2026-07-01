#!/usr/bin/env python3
"""Create RunPod serverless endpoint and network volume for worker-locate.

Steps:
  1. Create network volume (locate-models, 20GB, EU-RO-1) — idempotent via name check.
  2. Spin a temp agent pod to download the model to the volume.
  3. Create serverless endpoint (scale-to-zero, flashboot, RTX 4090, idleTimeout=300).
  4. Store endpoint id + volume id in earl keychain.

Usage:
  python3 deploy/provision.py [--skip-download] [--skip-endpoint]
  python3 deploy/provision.py --volume-id <existing> --endpoint-id <existing>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUNPOD_GQL = "https://api.runpod.io/graphql"
RUNPOD_REST = "https://api.runpod.io/v1"
DATACENTER = "EU-RO-1"
IMAGE = "ghcr.io/dr34mlab/worker-locate:latest"
VOLUME_NAME = "locate-models"
VOLUME_SIZE_GB = 20
ENDPOINT_NAME = "locate-anything"
IDLE_TIMEOUT = 300       # seconds — warm through a session
JOB_TIMEOUT = 120        # seconds
GPU_IDS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"]


def earl(account: str, optional: bool = False) -> str:
    try:
        return subprocess.check_output(
            ["security", "find-generic-password", "-s", "earl", "-a", account, "-w"],
            timeout=5,
        ).decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if optional:
            return ""
        raise


def store_earl(account: str, value: str) -> None:
    """Store or update a value in the earl keychain."""
    # Delete if exists, then add
    subprocess.run(
        ["security", "delete-generic-password", "-s", "earl", "-a", account],
        capture_output=True,
    )
    subprocess.check_call(
        ["security", "add-generic-password", "-s", "earl", "-a", account, "-w", value],
    )
    print(f"  stored earl key: {account}")


def gql(query: str, variables: dict | None = None, key: str = "") -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        RUNPOD_GQL,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def rest_post(path: str, body: dict, key: str) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{RUNPOD_REST}/{path}",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def ensure_volume(key: str) -> str:
    """Return existing locate-models volume id or create one."""
    resp = gql('query { myself { networkVolumes { id name size dataCenterId } } }', key=key)
    for v in resp["data"]["myself"]["networkVolumes"]:
        if v["name"] == VOLUME_NAME:
            print(f"  volume {VOLUME_NAME} exists: {v['id']} ({v['size']} GB)")
            return v["id"]

    print(f"  creating network volume {VOLUME_NAME} {VOLUME_SIZE_GB}GB in {DATACENTER} ...")
    resp = gql(
        """
        mutation CreateNetworkVolume($input: CreateNetworkVolumeInput!) {
          createNetworkVolume(input: $input) { id name size dataCenterId }
        }
        """,
        variables={"input": {
            "name": VOLUME_NAME,
            "size": VOLUME_SIZE_GB,
            "dataCenterId": DATACENTER,
        }},
        key=key,
    )
    vol = resp["data"]["createNetworkVolume"]
    print(f"  created volume: {vol['id']}")
    return vol["id"]


def download_model_to_volume(key: str, volume_id: str) -> None:
    """Spin a temporary agent pod to pull the HF model into the volume."""
    print("  creating temp download pod ...")
    script = Path(__file__).resolve().parent / "download_model.sh"
    cmd = "pip install huggingface_hub[hf_transfer] -q && " + script.read_text().replace("#!/usr/bin/env bash\n", "")

    # Use runpod/pytorch agent image (for runtime telemetry + SSH)
    resp = rest_post("pods", {
        "name": "locate-model-download",
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuTypeId": GPU_IDS[0],
        "cloudType": "SECURE",
        "dataCenterId": DATACENTER,
        "networkVolumeId": volume_id,
        "volumeMountPath": "/workspace",
        "dockerStartCmd": f"bash -c '{cmd}' && sleep 300",
        "env": [{"key": "HF_XET_HIGH_PERFORMANCE", "value": "1"}],
        "ports": "22/tcp",
    }, key=key)

    pod_id = resp.get("id", "")
    if not pod_id:
        print("  WARNING: pod create response:", resp)
        return

    print(f"  download pod id: {pod_id} — waiting for runtime (up to 10 min) ...")
    for i in range(60):
        time.sleep(10)
        info = gql(
            "query Pod($id: String!) { pod(input: {podId: $id}) { id runtime { uptimeInSeconds } } }",
            variables={"id": pod_id},
            key=key,
        )
        rt = info["data"]["pod"]["runtime"]
        if rt and rt.get("uptimeInSeconds", 0) > 0:
            print(f"  pod runtime up after ~{(i+1)*10}s")
            break
        if i % 6 == 5:
            print(f"  ... still waiting ({(i+1)*10}s)")
    else:
        print("  WARNING: pod never came up — model may not be downloaded")
        return

    # Give it time to download (model is ~6-7GB; Xet-fast = a few minutes)
    print("  Waiting 8 min for model download ...")
    time.sleep(480)

    # Terminate pod
    gql(
        "mutation TerminatePod($input: PodTerminateInput!) { podTerminate(input: $input) }",
        variables={"input": {"podId": pod_id}},
        key=key,
    )
    print("  download pod terminated.")


def create_endpoint(key: str, volume_id: str) -> str:
    """Create serverless endpoint and return its id."""
    print("  creating serverless endpoint ...")
    resp = gql(
        """
        mutation SaveTemplate($input: SaveTemplateInput!) {
          saveTemplate(input: $input) { id name }
        }
        """,
        variables={"input": {
            "name": "worker-locate",
            "imageName": IMAGE,
            "isPublic": False,
            "isRunPodOfficial": False,
            "env": [
                {"key": "LA_MODEL_PATH", "value": "/runpod-volume/models/locate-anything"},
                {"key": "PYTHONUNBUFFERED", "value": "1"},
            ],
            "containerDiskInGb": 10,
            "volumeInGb": 0,
        }},
        key=key,
    )
    template_id = resp["data"]["saveTemplate"]["id"]
    print(f"  template id: {template_id}")

    resp = gql(
        """
        mutation SaveEndpoint($input: EndpointInput!) {
          saveEndpoint(input: $input) { id name }
        }
        """,
        variables={"input": {
            "name": ENDPOINT_NAME,
            "templateId": template_id,
            "gpuIds": ",".join(GPU_IDS),
            "networkVolumeId": volume_id,
            "idleTimeout": IDLE_TIMEOUT,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
            "workersMin": 0,
            "workersMax": 3,
            "flashboot": True,
            "locations": DATACENTER,
            "jobTimeout": JOB_TIMEOUT,
        }},
        key=key,
    )
    ep_id = resp["data"]["saveEndpoint"]["id"]
    print(f"  endpoint id: {ep_id}")
    return ep_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true", help="Skip model download to volume")
    ap.add_argument("--skip-endpoint", action="store_true", help="Skip endpoint creation")
    ap.add_argument("--volume-id", default="", help="Use existing volume id")
    ap.add_argument("--endpoint-id", default="", help="Use existing endpoint id (just store in earl)")
    args = ap.parse_args()

    key = earl("RUNPOD_API_KEY_CLI")

    # Volume
    volume_id = args.volume_id or ensure_volume(key)

    # Model download
    if not args.skip_download:
        download_model_to_volume(key, volume_id)
    else:
        print("  skipping model download.")

    # Endpoint
    if args.endpoint_id:
        ep_id = args.endpoint_id
        print(f"  using existing endpoint: {ep_id}")
    elif not args.skip_endpoint:
        ep_id = create_endpoint(key, volume_id)
    else:
        ep_id = ""
        print("  skipping endpoint creation.")

    # Store in earl
    store_earl("runpod_locate_volume_id", volume_id)
    if ep_id:
        store_earl("runpod_locate_endpoint_id", ep_id)

    print()
    print("Done.")
    print(f"  volume_id:   {volume_id}")
    print(f"  endpoint_id: {ep_id}")
    print()
    print(f"  Health: curl -H 'Authorization: Bearer $RUNPOD_API_KEY_CLI' https://api.runpod.ai/v2/{ep_id}/health")


if __name__ == "__main__":
    main()
