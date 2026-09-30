"""
convert_to_onnx.py — Run this ONCE on any machine (Windows/Linux/Mac)
before copying models to the Raspberry Pi.

Exports every .pt model in models/ to .onnx (FP32, opset 17).
The Pi will auto-detect and load the .onnx via ONNX Runtime — no
PyTorch required at inference time, giving ~2-3x speedup on Cortex-A72.

Usage:
    python convert_to_onnx.py
    python convert_to_onnx.py --imgsz 320   # smaller = faster on Pi
    python convert_to_onnx.py --models models/appearance.pt models/behavior.pt
"""
from __future__ import annotations

import argparse
import os
import sys


def export_model(pt_path: str, imgsz: int) -> str:
    from ultralytics import YOLO
    print(f"  Loading {pt_path} ...")
    model = YOLO(pt_path)
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        simplify=True,
        dynamic=False,
        half=False,          # FP32 — ONNX Runtime on arm64 handles FP32 natively
        device="cpu",
    )
    print(f"  Exported → {out}")
    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Export .pt models to .onnx for Raspberry Pi")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Inference image size (default 416, use 320 for max Pi speed)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Paths to .pt files. Defaults to all .pt in models/")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    if args.models:
        pt_files = args.models
    else:
        if not os.path.isdir(base):
            print(f"[error] models/ directory not found at {base}")
            sys.exit(1)
        pt_files = [
            os.path.join(base, f)
            for f in os.listdir(base)
            if f.endswith(".pt")
        ]

    if not pt_files:
        print("[warn] No .pt files found.")
        sys.exit(0)

    print(f"[convert] imgsz={args.imgsz}  models={len(pt_files)}\n")
    for pt in pt_files:
        if not os.path.isfile(pt):
            print(f"  [skip] {pt} — not found")
            continue
        try:
            export_model(pt, args.imgsz)
        except Exception as exc:
            print(f"  [error] {pt}: {exc}")

    print("\n[done] Copy the .onnx files alongside .pt in models/ on the Pi.")
    print("       The predictor auto-loads .onnx when onnxruntime is installed.")


if __name__ == "__main__":
    main()
