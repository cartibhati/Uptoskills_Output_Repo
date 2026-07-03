# Task 131: AI-Based Crowd Congestion Detection System

## Project Overview
This project implements a modular **AI-Based Crowd Congestion Detection System** for public safety and traffic analytics. It integrates:
1. **Pretrained YOLOv8 Object Detection** to identify and locate people (class 0).
2. **Lightweight Intersection-over-Union (IoU) Tracking** to track persons across frames and avoid double counting.
3. **Spatial Zone Management** utilizing polygon point-in-polygon checks (`cv2.pointPolygonTest`) to map tracked people's footprints to predefined regions.
4. **Crowd Density Estimation** per zone with configurable thresholds mapping crowd counts to LOW, MODERATE, HIGH, and CRITICAL levels.
5. **Congestion Hotspot Detection** using a temporal persistence state machine to identify sustained overcrowding and transition zone status from NORMAL to CONGESTED.
6. **Pluggable Alerts & CSV Logs** logging transition events, crowd count telemetry, and saving density heatmaps.

---

## Folder Structure

```
Task131_CrowdCongestionDetectionSystem/
├── config.yaml               # System parameters, zone definitions, thresholds, paths
├── requirements.txt         # Project dependencies
├── README.md                # This instructions file
├── Code/
│   ├── main.py                  # Pipeline orchestrator & command-line entrypoint
│   ├── detector.py              # YOLOv8 Person detector wrapper (pretrained, class 0 only)
│   ├── tracker.py               # Custom lightweight tracking logic (SimpleIoUTracker)
│   ├── zone_manager.py          # Point-in-polygon assignment, density estimation, and state machine
│   ├── alert_manager.py         # Persistent alert logging & pluggable notification delivery
│   ├── heatmap_generator.py     # Centroid density accumulator and heatmap renderer
│   ├── output_writer.py         # Video writer & CSV telemetry writer
│   ├── utils.py                 # Shared drawing and geometry helper functions
│   ├── crowd_congestion_detection_system.ipynb  # Modular import-based notebook
│   └── Task131_Standalone.ipynb                 # Standalone, single-notebook execution
├── Outputs/                 # Output directory for annotated videos, CSV logs, and heatmaps
├── Models/                  # Output directory for YOLOv8 model weights
└── Test_Videos/             # Directory containing sample videos for testing
```

---

## Setup & Dependencies

### 1. Python Environment
Ensure you have Python 3.8+ installed. Install the required python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Test Video (Manual Setup)
The sample video used for testing (`Test_Videos/sample.mp4`) is not included in the repository. Before running the pipeline, please supply a test video at the `Test_Videos/sample.mp4` path (or update `video_path` in `config.yaml` to point to your video).

### 3. YOLOv8 Model Weights
The pretrained YOLOv8n weights (`Models/yolov8n.pt`) are excluded from the repository. On the first execution, the pipeline's detector wrapper will automatically check if the model exists locally. If not found, it will automatically download `yolov8n.pt` from the Ultralytics servers.

---

## Configuration (`config.yaml`)

The system parameters are defined in `config.yaml`. The key parameters include:
*   `video_path`: Path to the video sequence.
*   `output_dir`: Path where output annotated video, CSV logs, and heatmaps are written.
*   `model_path`: Path to YOLOv8 weights (e.g. `Models/yolov8n.pt`).
*   `detector.confidence_threshold`: Minimum confidence score for detection.
*   `congestion.min_duration_seconds`: Dwell time threshold in seconds for a zone to transition to a CONGESTED state.
*   `zones`: List of spatial zones defined by a name, bounding polygon vertices (in `[x, y]` format), and count thresholds for density level mapping.

---

## How to Run

### 1. Command-Line Execution
Run the system on the default video specified in `config.yaml`:
```bash
python Code/main.py --config config.yaml
```

You can override the config file, video path, or output directory via CLI arguments:
```bash
python Code/main.py --config config.yaml --video Test_Videos/sample.mp4 --output_dir Outputs
```

### 2. Interactive Development Notebook (`Code/crowd_congestion_detection_system.ipynb`)
Open `Code/crowd_congestion_detection_system.ipynb` in Jupyter Notebook/JupyterLab. This notebook imports the modular `.py` files from the codebase and runs the detection pipeline step-by-step, allowing interactive parameter adjustments and visualization.

### 3. Fully Self-Contained Notebook (`Code/Task131_Standalone.ipynb`)
For testing in cloud environments (like Google Colab) or environments without local scripts, run `Code/Task131_Standalone.ipynb`. This notebook contains all classes and functions defined directly in cells with zero dependencies on other `.py` files. It renders inline visualizations of:
*   Sample video frames.
*   Live telemetry curves (counts over time).
*   The final density heatmap.

---

## Expected Outputs

Upon completion of a pipeline run, the following files will be created in the `Outputs/` directory:
1.  **Annotated Video (`Outputs/<video_name>_annotated.mp4`)**:
    *   Detected persons annotated with bounding boxes and tracking IDs.
    *   Spatial zone polygons overlaid and color-coded by density: Green (LOW), Yellow (MODERATE), Orange (HIGH), Red (CRITICAL/CONGESTED).
    *   Active HUD overlay displaying live counts and density states.
    *   Flashing top banner alert when congestion is detected.
2.  **Telemetry Data (`Outputs/crowd_density.csv`)**:
    *   Granular per-frame telemetry: `timestamp, frame_number, zone, crowd_count, congestion_level, is_congested`.
3.  **Alerts Log (`Outputs/alerts.log`)**:
    *   Event-based audit log capturing congestion start and resolution details: `timestamp, zone, event_type (START/END), count, duration_seconds`.
4.  **Density Heatmaps (`Outputs/heatmap_final.png` & snapshots)**:
    *   A smoothed (Gaussian blurred) density heatmap colormapped (`cv2.COLORMAP_JET`) and overlaid on the reference video frame.
5.  **Summary Statistics (`Outputs/summary_stats.json`)**:
    *   Overall execution summary including processed frames, execution time, average FPS, and details of all congestion events.
