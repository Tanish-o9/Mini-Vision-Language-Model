"""
Mini-VLM Instant Public Sharing Helper
Creates a free, zero-config Cloudflare HTTPS tunnel to expose http://localhost:8000
to any device or person across the internet without port-forwarding or account sign-in.
"""
import os
import re
import sys
import shutil
import urllib.request
import subprocess
from pathlib import Path

TOOLS_DIR = Path(".tools")
CLOUDFLARED_EXE = TOOLS_DIR / "cloudflared.exe"
DOWNLOAD_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

def ensure_cloudflared():
    if shutil.which("cloudflared"):
        return "cloudflared"
    if CLOUDFLARED_EXE.exists():
        return str(CLOUDFLARED_EXE)
    
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading lightweight Cloudflare Tunnel client (first-time only, ~15MB)...")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, str(CLOUDFLARED_EXE))
        print("Download complete!")
        return str(CLOUDFLARED_EXE)
    except Exception as e:
        print(f"Automatic download failed: {e}")
        return None

def start_tunnel(port=8000):
    bin_path = ensure_cloudflared()
    if not bin_path:
        print("\nTip: You can also use ngrok (ngrok http 8000) or localtunnel (npx localtunnel --port 8000).")
        return

    print("=" * 65)
    print(f"Starting Secure Public HTTPS Tunnel for Mini-VLM on port {port}...")
    print("=" * 65)

    cmd = [bin_path, "tunnel", "--url", f"http://localhost:{port}"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    tunnel_url = None
    for line in iter(process.stdout.readline, ""):
        # Search for *.trycloudflare.com URL
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            tunnel_url = match.group(0)
            print("\n" + "*" * 65)
            print(f"  🚀 YOUR LIVE PUBLIC DEMO URL IS READY:")
            print(f"     {tunnel_url}")
            print("*" * 65)
            print("Share this link with anyone to test Mini-VLM from their phone or PC!\n")
            print("Press Ctrl+C to stop sharing.\n")
            break

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping tunnel...")
        process.terminate()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_tunnel(port)

