"""
test_image.py — run the exact same detection pipeline used by the live
camera feed against a single image file. Use this to sanity-check the
models against known example photos BEFORE wiring up the USB camera —
much faster to iterate on than holding a phone up to a webcam.

Usage:
    python test_image.py path/to/photo.jpg
    python test_image.py path/to/photo.jpg --out annotated.jpg
"""
from __future__ import annotations

import argparse
import sys

import cv2

from predictor import PoultryPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a poultry image to test")
    parser.add_argument(
        "--out", default="annotated.jpg",
        help="Where to save the annotated result (default: annotated.jpg)",
    )
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}")
        sys.exit(1)

    predictor = PoultryPredictor(debug=True)
    res = predictor.predict_uploaded(frame)
    det = res.primary

    if det is None:
        print("\nNo confident detection from any model on this image.")
        return

    print(f"\nAgent Domain:  {res.segregation_context}")
    print(f"Agent Reason:  {res.segregation_reason}")
    print(f"Prediction:    {det.label}")
    print(f"Confidence:    {det.confidence:.1%}")
    print(f"Source model:  {det.source_model}  (raw class: {det.raw_label!r})")

    predictor.annotate(frame, det)
    cv2.imwrite(args.out, frame)
    print(f"Annotated image saved to: {args.out}")


if __name__ == "__main__":
    main()
