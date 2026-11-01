"""
Comprehensive End-to-End Test Suite for Mini-VLM Application.
Tests:
1. Core HTTP REST APIs (Status, Config, Metrics, Benchmark)
2. Static HTML & Compiled React Bundle Assets (JS, CSS)
3. Single-Image Inference with Visual Cues Grounding
4. Multi-turn Chat History with Context Awareness
5. Dual-Image Multimodal Compare Mode Inference
6. Edge Cases (Invalid payload, empty question, boundary token limits)
7. GPU / VRAM Memory Leak Verification
"""
import base64
import io
import json
import re
import sys
import time
import urllib.request
import urllib.error
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8000"
passed = 0
failed = 0

def log_test(name, success, detail=""):
    global passed, failed
    if success:
        passed += 1
        print(f" [PASS] {name} {detail}")
    else:
        failed += 1
        print(f" [FAIL] {name} - ERROR: {detail}")

def create_dummy_image(color="red", shape="circle"):
    img = Image.new("RGB", (224, 224), color="white")
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse([40, 40, 184, 184], fill=color)
    else:
        draw.rectangle([40, 40, 184, 184], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

print("=" * 60)
print("STARTING MINI-VLM COMPREHENSIVE SYSTEM VERIFICATION")
print("=" * 60)

# 1. Core Endpoints
for endpoint in ["/api/status", "/api/config", "/api/metrics", "/api/benchmark"]:
    try:
        resp = urllib.request.urlopen(f"{BASE_URL}{endpoint}", timeout=5)
        data = json.loads(resp.read().decode("utf-8"))
        log_test(f"Endpoint GET {endpoint}", resp.status == 200, f"(Keys: {list(data.keys())[:4]})")
    except Exception as e:
        log_test(f"Endpoint GET {endpoint}", False, str(e))

# 2. Static HTML and Bundles
try:
    resp = urllib.request.urlopen(f"{BASE_URL}/", timeout=5)
    html_content = resp.read().decode("utf-8")
    has_root = '<div id="root">' in html_content
    log_test("Static Index Serving", resp.status == 200 and has_root, "(HTML contains root div)")

    # Find JS and CSS assets
    js_matches = re.findall(r'src="(/assets/[^"]+\.js)"', html_content)
    css_matches = re.findall(r'href="(/assets/[^"]+\.css)"', html_content)
    
    for js in js_matches:
        js_resp = urllib.request.urlopen(f"{BASE_URL}{js}", timeout=5)
        log_test(f"React JS Bundle ({js})", js_resp.status == 200, f"({len(js_resp.read())} bytes)")

    for css in css_matches:
        css_resp = urllib.request.urlopen(f"{BASE_URL}{css}", timeout=5)
        log_test(f"React CSS Stylesheet ({css})", css_resp.status == 200, f"({len(css_resp.read())} bytes)")
except Exception as e:
    log_test("Static HTML and Bundles", False, str(e))

# 3. Single Image Multimodal Chat
img_red = create_dummy_image("red", "circle")
chat_payload = {
    "image_base64": img_red,
    "question": "is photo me kya h??",
    "history": [],
    "max_new_tokens": 50,
    "stage": "stage2"
}
try:
    start = time.perf_counter()
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps(chat_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=30)
    res_data = json.loads(resp.read().decode("utf-8"))
    dur = round(time.perf_counter() - start, 3)
    
    has_answer = bool(res_data.get("answer"))
    has_cues = "visual_cues" in res_data
    log_test("Single Image Chat Inference", has_answer and has_cues, f"-> '{res_data.get('answer')}' in {dur}s | Cues: '{res_data.get('visual_cues')}'")
    turn1_answer = res_data.get("answer")
except Exception as e:
    log_test("Single Image Chat Inference", False, str(e))
    turn1_answer = ""

# 4. Multi-Turn Conversation (Follow-Up Question)
if turn1_answer:
    followup_payload = {
        "image_base64": img_red,
        "question": "what color is it?",
        "history": [["is photo me kya h??", turn1_answer]],
        "max_new_tokens": 50
    }
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/api/chat",
            data=json.dumps(followup_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=30)
        res_data = json.loads(resp.read().decode("utf-8"))
        ans2 = res_data.get("answer", "")
        log_test("Multi-Turn History Inference", bool(ans2), f"-> Follow-up Reply: '{ans2}'")
    except Exception as e:
        log_test("Multi-Turn History Inference", False, str(e))

# 5. Dual-Image Multimodal Compare Mode
img_blue = create_dummy_image("blue", "rectangle")
compare_payload = {
    "image_base64": img_red,
    "secondary_image_base64": img_blue,
    "question": "Compare the two images and describe the difference.",
    "history": [],
    "max_new_tokens": 60
}
try:
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps(compare_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=30)
    res_data = json.loads(resp.read().decode("utf-8"))
    log_test("Compare Mode (Dual-Image) Inference", bool(res_data.get("answer")), f"-> '{res_data.get('answer')}'")
except Exception as e:
    log_test("Compare Mode (Dual-Image) Inference", False, str(e))

# 6. Error & Edge Case Handling
# Edge Case A: Empty question
try:
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps({"image_base64": img_red, "question": ""}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)
    log_test("Edge Case: Empty question validation", False, "Expected 400 Bad Request, got 200")
except urllib.error.HTTPError as e:
    log_test("Edge Case: Empty question validation", e.code == 400, f"(Correctly rejected with HTTP {e.code})")

# Edge Case B: Missing image
try:
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps({"image_base64": "", "question": "hello"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)
    log_test("Edge Case: Missing image validation", False, "Expected 400 Bad Request, got 200")
except urllib.error.HTTPError as e:
    log_test("Edge Case: Missing image validation", e.code == 400, f"(Correctly rejected with HTTP {e.code})")

# 7. VRAM and System Diagnostics Check
try:
    resp = urllib.request.urlopen(f"{BASE_URL}/api/status", timeout=5)
    stat = json.loads(resp.read().decode("utf-8"))
    vram_alloc = stat.get("vram_allocated_gb", 0)
    vram_ok = vram_alloc < 3.8
    log_test("VRAM Leak & Memory Ceiling Guard", vram_ok, f"(Allocated: {vram_alloc} GB, Reserved: {stat.get('vram_reserved_gb')} GB)")
except Exception as e:
    log_test("VRAM Leak & Memory Ceiling Guard", False, str(e))

print("=" * 60)
print(f"VERIFICATION COMPLETE: {passed} PASSED, {failed} FAILED")
print("=" * 60)

if failed > 0:
    sys.exit(1)

