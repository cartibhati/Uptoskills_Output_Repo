# Task 130: AI-Based Fall Detection System

## Task Overview
This project implements a rule-based **Fall Detection System** for public safety surveillance. It integrates:
1. **YOLOv8-Pose** for human joint extraction (extracting bounding boxes and 17 COCO pose keypoints).
2. **IoU Tracker** to match person IDs across frames.
3. **Physical Heuristics** (Torso Angle vs Vertical Axis, Aspect Ratio, and Centroid Velocity) with a **temporal persistence guard (1.0 second)** to distinguish actual falls from Activities of Daily Living (ADLs) like crouching or bending down.
4. **Alerts & Logging** that triggers a console warning, logs structured data to a CSV index, and saves an evidence snapshot when a person transitions to a fallen state.

---

## Folder Architecture & Alignment
In accordance with the team guidelines, the project structure is organized as follows:

```
Task130_FallDetectionSystem/
├── Code/
│   ├── detect.py                   # Command-line pipeline execution script
│   ├── utils.py                    # Helper module (tracking, heuristics, visualization, dashboard)
│   └── fall_detection_system.ipynb # Restructured Jupyter Notebook for Google Colab
├── Outputs/
│   ├── fall_test_annotated.mp4     # Processed/Annotated output video for Fall Test 1
│   ├── fall_test_urfd_annotated.mp4# Processed/Annotated output video for Fall Test 2 (URFD dataset)
│   ├── crouch_test_annotated.mp4   # Processed/Annotated negative control video (crouch test)
│   ├── incident_log.csv            # Structured pandas DataFrame CSV log of incidents
│   ├── summary_stats.json          # Overall summary statistics JSON
│   ├── analytics_dashboard.png     # Rendered matplotlib dashboard plots
│   └── evidence_frames/            # Evidence snapshot JPG files saved at fall transition point
│       ├── incident_20260626_164103_2.jpg
│       └── incident_20260626_164136_1.jpg
└── Models/
    └── yolov8n-pose.pt             # YOLOv8 Pose model weights (6.8 MB)
```

---

## Verification & Execution Outcomes

We verified the pipeline on **3 actual video files** representing diverse movements. Below are the execution results:

### 1. Video 1: Standard Fall (`fall_test.mp4`)
*   **Properties**: 640x360 pixels | 23.98 FPS | 272 frames
*   **Outcome**: **Fall Detected** at Frame 130 (Person ID 2).
*   **Log**: Saved `Outputs/evidence_frames/incident_20260626_164103_2.jpg`.
*   **Performance**: 16.56 FPS average processing speed.

### 2. Video 2: UR Fall Detection Dataset (`fall_test_urfd.mp4`)
*   **Properties**: 1280x720 pixels | 30.00 FPS | 234 frames
*   **Outcome**: **Fall Detected** at Frame 217 (Person ID 1).
*   **Log**: Saved `Outputs/evidence_frames/incident_20260626_164136_1.jpg`.
*   **Performance**: 14.92 FPS average processing speed.

### 3. Video 3: Crouch/Negative Control (`crouch_test.mp4`)
*   **Properties**: 768x432 pixels | 12.00 FPS | 596 frames
*   **Outcome**: **No Fall Detected (Normal)**. The aspect ratio and temporal persistence check successfully filtered out crouching/bending actions, resulting in 0 false positive alerts.
*   **Performance**: 19.29 FPS average processing speed.

---

## How to Run

### Command Line Execution
To run the detection script on any video:
```bash
python Task130_FallDetectionSystem/Code/detect.py --video <path_to_video>
```

### Google Colab Notebook
Teammates can upload the `Task130_FallDetectionSystem/Code/fall_detection_system.ipynb` to Google Colab, mount Google Drive, and run the cells sequentially to test their own videos.
