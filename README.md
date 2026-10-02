# Outfit studio and posture mapping

Set up your Iphone(running ios 18 or greater), run outfit studio. Press the space bar and pose. Review all your saved outfits and pose data later.

A Python application that uses **OpenCV** to capture video feeds (utilizing apples continuity camera) and **Ultralytics YOLO** for real-time pose detection.

## Prerequisites

Before running this project, ensure you have **Python 3.8 or higher** installed on your system.

## Getting Started

You can install and use Outfit Studio in two different ways: as a quick tool via PyPI, or by cloning the source code for local development. First time run will download yolov8 pose model.

### Option A: Quick Installation (via PyPI)

If you just want to use the tool, install it directly using `pip`:

```bash
pip install outfit-studio
```

Once installed, launch the application instantly from anywhere in your terminal:

```bash
outfitstudio-start
```

---

### Option B: Local Development (via GitHub)

If you want to modify the source code or run it locally from the repository, follow these steps:

#### 1. Clone the Repository

```bash
git clone https://github.com
cd outfit-studio
```

#### 2. Install Package Dependencies

Install the required packages directly from the root folder:

```bash
pip install -r requirements.txt
```

#### 3. Run the Local Source

You can start the script from the directory containing your package code:

```bash
python outfitstudio/studio.py
```

_(Note: On its first run, the system will automatically download the required YOLO pose weights file into your project folder)._

---

## How to Control the Application

Once the camera window appears on your screen:

- **Spacebar** - Captures a snapshot (incorporates a 1-second delay so you can pose).
- **ESC Key** - Safely closes the application stream and windows.

Your captured images are split and automatically organized inside an auto-generated `outfit_output/` folder structure:

- `outfit_output/clean/` - The original high-quality raw photo.
- `outfit_output/labeled/` - The photo overlaid with your skeleton tracking dots and your overall pose accuracy score.

---

## Core Dependencies

- **[Ultralytics YOLO](https://github.com)** - Deep learning vision framework utilized for pose tracking.
- **[OpenCV](https://github.com)** - Multi-platform framework used for capturing video feeds and burning graphic overlays.
