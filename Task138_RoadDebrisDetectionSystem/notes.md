# Task 138: AI-Based Road Debris & Obstacle Detection System

## Task Overview

This project implements a robust **Road Debris & Obstacle Detection System** to detect, track, and log hazards present in roadway lanes. It integrates:
1. **YOLOv8** for real-time object detection (identifying standard vehicles/pedestrians and debris items like suitcases, boxes, or chairs).
2. **Class-Aware IoU Tracker** to assign persistent IDs and compile centroid histories.
3. **Road Region of Interest (ROI) Restriction** to filter out off-road objects and focus analysis purely on the drivable lane (defaulting to a normalized trapezoidal zone).
4. **Dual Camera Motion Heuristics**:
   - **Stationary Camera Mode (`stationary`)**: Monitors highway/traffic surveillance feeds. It checks for absolute centroid immobility (displacement below 1.5% of width) and filters out moving traffic using persistence thresholds (1.5s for debris, 4.0s for vehicles).
   - **Moving Camera Mode (`moving`)**: Monitors vehicle dashcams. Bypasses absolute immobility (since the camera itself moves) and instead tracks the relative vertical velocity ($v_y$) of objects. Objects moving downwards rapidly relative to the camera ($v_y \geq 0.002 \times \text{height}$) inside the road ROI are flagged as approaching hazards with low persistence (0.2s = 5 frames).
5. **Alerts & Logging** that write warning alerts, log incident metadata to a CSV log, save evidence snapshots of obstacles, and compile overall statistics.
6. **Analytics Dashboard** which visualizes class distributions and incident timelines.

---

## Folder Architecture & Alignment

In accordance with internship guidelines, the project workspace is structured as follows:

```text
Task138_RoadDebrisDetectionSystem/
├── Code/
│   ├── detect.py                   # Standalone CLI execution script
│   ├── utils.py                    # Helper module (frame reader, tracker, logging, dashboarding)
│   └── road_debris_detection_system.ipynb # Restructured Jupyter Notebook for Google Colab
├── Inputs/
│   ├── highway.mp4                 # Test Video 1: Normal traffic flow (768x432, 12 FPS, 647 frames)
│   ├── trash.mp4                   # Test Video 2: Active road debris hazard - moving dashcam (640x340, 24 FPS, 291 frames)
│   └── large_debris.mp4            # Test Video 3: Active road debris hazard - stationary surveillance (1920x1080, 30 FPS, 130 frames)
├── Models/
│   └── yolov8n.pt                  # Pre-trained YOLOv8 weights (6.2 MB)
├── Outputs/
│   ├── highway_annotated.mp4       # Processed output video for Test 1 (0 alerts)
│   ├── trash_annotated.mp4         # Processed output video for Test 2 (1 alert)
│   ├── large_debris_annotated.mp4 # Processed output video for Test 3 (1 alert)
│   ├── incident_log.csv            # Structured CSV database log of incidents (2 rows)
│   ├── summary_stats.json          # Overall summary statistics JSON
│   ├── analytics_dashboard.png     # Rendered matplotlib dashboard plots
│   └── evidence_frames/            # JPG snapshots saved at debris transition points
│       ├── incident_20260705_143703_car_id_6.jpg
│       └── incident_20260705_160609_car_id_1.jpg
└── notes.md                        # Task notes and outcomes log (this file)
```

---

## Verification & Execution Outcomes

We verified the pipeline on **3 test videos** representing different camera setups:

### 1. Video 1: Normal Highway Traffic (`highway.mp4`) - Negative Control
*   **Properties**: 768x432 pixels | 12.00 FPS | 647 frames
*   **Execution Command**:
    ```bash
    python Task138_RoadDebrisDetectionSystem/Code/detect.py --video Task138_RoadDebrisDetectionSystem/Inputs/highway.mp4 --camera_motion stationary
    ```
*   **Outcome**: **0 incidents detected**. Moving vehicles in the lanes were tracked but correctly classified as moving. The stationary filter successfully prevented false positive alerts.
*   **Performance**: ~20.86 FPS average processing speed.

### 2. Video 2: Active Road Debris Hazard (`trash.mp4`) - Positive Control (Moving Camera)
*   **Properties**: 640x340 pixels | 24.00 FPS | 291 frames
*   **Execution Command**:
    ```bash
    python Task138_RoadDebrisDetectionSystem/Code/detect.py --video Task138_RoadDebrisDetectionSystem/Inputs/trash.mp4 --camera_motion moving
    ```
*   **Outcome**: **1 incident detected**. The road debris (a large trash object in the middle of our lane) was detected (ID 6, misclassified as `car` with 0.61 confidence) and triggered an alert at Frame 89.
*   **Log**: Saved `Outputs/evidence_frames/incident_20260705_143703_car_id_6.jpg`.
*   **Performance**: ~18.22 FPS average processing speed.

### 3. Video 3: Active Road Debris Hazard (`large_debris.mp4`) - Positive Control (Stationary Camera)
*   **Properties**: 1920x1080 pixels | 30.00 FPS | 130 frames
*   **Execution Command**:
    ```bash
    python Task138_RoadDebrisDetectionSystem/Code/detect.py --video Task138_RoadDebrisDetectionSystem/Inputs/large_debris.mp4 --camera_motion stationary
    ```
*   **Outcome**: **1 incident detected**. The large trash container directly blocking the lane was tracked from Frame 0. After remaining stationary for 120 frames (4.0 seconds), it triggered an alert at Frame 123 (ID 1, classified as `car` with 0.81 confidence).
*   **Log**: Saved `Outputs/evidence_frames/incident_20260705_160609_car_id_1.jpg`.
*   **Performance**: ~8.72 FPS average processing speed.

### 4. Video 4: Overturned Vehicle Hazard (`overturn_0.mp4`) - Positive Control (Moving Camera, Far Distance)
*   **Properties**: 1920x1080 pixels | 30.00 FPS | 145 frames
*   **Execution Command**:
    ```bash
    python Task138_RoadDebrisDetectionSystem/Code/detect.py --video Task138_RoadDebrisDetectionSystem/Inputs/overturn_0.mp4 --roi "0.35,0.48,0.45,0.25,0.55,0.25,0.65,0.48" --camera_motion moving --speed_threshold 0.0001
    ```
*   **Outcome**: **2 incidents detected**. The custom Road ROI was adjusted to focus on the distant road lane and exclude the ego-vehicle and cafe tables on the side. The overturned vehicle was detected as both `car` (ID 6, confidence 0.43) and `truck` (ID 2, confidence 0.35) and successfully triggered warning alerts (at frame 32 and 41 respectively) due to its relative forward approach velocity.
*   **Log**: Saved `Outputs/evidence_frames/incident_20260706_140759_car_id_6.jpg` and `Outputs/evidence_frames/incident_20260706_140759_truck_id_2.jpg`.
*   **Performance**: ~8.41 FPS average processing speed.

---

## How to Run

### Standalone CLI Execution
To execute the pipeline:
```bash
# To run in stationary camera mode (default)
python Task138_RoadDebrisDetectionSystem/Code/detect.py --video <path_to_video>

# To run in moving camera mode (dashcam)
python Task138_RoadDebrisDetectionSystem/Code/detect.py --video <path_to_video> --camera_motion moving
```

### Google Colab Notebook
Upload the `Task138_RoadDebrisDetectionSystem/Code/road_debris_detection_system.ipynb` to Google Colab, mount Google Drive, place video files under `Inputs/`, and execute the cells sequentially.
