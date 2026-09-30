# Poultry Disease Prediction

A Flask-based computer-vision application for screening poultry health from a USB-camera stream or uploaded image. The application uses ONNX/YOLO models to identify named diseases, report uncertain cases as abnormal, and show detection confidence with annotations.

> This project is a screening aid, not a veterinary diagnosis. Isolate birds with suspected illness and consult a qualified veterinarian or local animal-health authority.

## Features

- Live USB-camera dashboard with annotated detections
- Image-upload prediction endpoint and command-line image tester
- Four model modes: `head`, `fecal`, `triage`, and `ensemble`
- Detection of Fowl Pox, Favus, Infectious Coryza, CRD, Weak Leg Syndrome, Bumblefoot, Coccidiosis, Newcastle Disease, and Salmonellosis
- Healthy, abnormal, and unidentified-condition results when a specific disease is not confidently identified
- ONNX Runtime inference for efficient deployment, with `.pt` model fallback
- Camera reconnect, camera-health, prediction-mode, and confidence-threshold controls

## Project structure

```text
Poultry_disease_Prediction/
|- app.py                     Flask web application and API routes
|- camera.py                  Threaded camera capture and live inference
|- predictor.py               Model loading, prediction, and result aggregation
|- test_image.py              Predict a single image from the command line
|- convert_to_onnx.py         Export compatible PyTorch models to ONNX
|- requirements.txt           Python dependencies
|- Object_Detection_models/   ONNX and PyTorch detection-model weights
|- classification_models/     Supplementary classification-model assets
|- templates/index.html       Web dashboard
|- static/                    Dashboard images and animations
`- prediction_images/         Example images for testing
```

## Requirements

- Python 3.10 or later
- USB camera for live detection (optional for image uploads)
- Model files in `Object_Detection_models/`

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source venv/bin/activate
```

## Run the web application

```powershell
python app.py
```

Open `http://localhost:5000` in a browser. To access it from another device on the same network, open `http://<computer-ip>:5000`.

### Optional environment variables

```powershell
# Select a camera index (default: 0)
$env:CAMERA_SOURCE = "1"

# Select the initial prediction mode (default: ensemble)
$env:DEFAULT_MODE = "head"

python app.py
```

Available prediction modes:

| Mode | Models used | Intended input |
| --- | --- | --- |
| `head` | `general` | Head, comb, skin, leg, or body symptoms |
| `fecal` | `fecal` | Droppings or fecal samples |
| `triage` | `appearance` and `behavior` | General healthy/abnormal screening |
| `ensemble` | All four models | Broad screening across supported inputs |

## Test one image

```powershell
python test_image.py .\prediction_images\image.jpg
```

The command prints the result and saves an annotated output image (by default, `annotated.jpg`).

## API overview

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Dashboard |
| `/video_feed` | `GET` | Live MJPEG camera stream |
| `/status` | `GET` | Latest prediction result |
| `/predict_image` | `POST` | Predict an uploaded image (`image` form field) |
| `/set_mode` | `POST` | Change prediction mode |
| `/set_threshold` | `POST` | Change confidence threshold |
| `/reconnect_camera` | `POST` | Reconnect or switch camera |
| `/camera_health` | `GET` | Camera status |

## Raspberry Pi deployment

For Raspberry Pi OS Bookworm setup, ONNX export, and optional auto-start service instructions, see [SETUP.md](SETUP.md).

## Notes

- ONNX files are preferred at runtime for lower resource usage; the app can fall back to PyTorch weights when required dependencies are available.
- Image quality, lighting, framing, and similarity between diseases can affect predictions. Treat low-confidence or abnormal results as prompts for closer examination, not confirmation of disease.
