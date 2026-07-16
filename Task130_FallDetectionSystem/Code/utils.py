import os
import cv2
import math
import glob
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# SKELETON DEFINITION FOR VISUALIZATION
# =====================================================================
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Head/Face
    (5, 6),                              # Shoulders midpoint link
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12),                    # Torso borders
    (11, 12),                            # Hips midpoint link
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16)                   # Right leg
]

def video_frame_generator(video_path):
    """
    Opens a video file OR a directory containing sequential images, and yields frames 
    sequentially along with sequence metadata.
    
    This function uses try/except blocks to ensure corrupted or invalid files 
    print a clear error message instead of crashing.
    
    Args:
        video_path (str): Path to the input video file or image sequence directory.
        
    Yields:
        tuple: (frame, frame_idx, fps, frame_count, width, height)
            - frame (numpy.ndarray): The current video frame.
            - frame_idx (int): The current frame index (0-based).
            - fps (float): Frame rate of the video.
            - frame_count (int): Total number of frames in the sequence.
            - width (int): Width of the video frame.
            - height (int): Height of the video frame.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input path not found at: {video_path}")
        
    # Check if the path is a directory of images (e.g. frame-by-frame datasets)
    if os.path.isdir(video_path):
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(video_path, ext)))
            image_files.extend(glob.glob(os.path.join(video_path, ext.upper())))
            
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in directory: {video_path}")
            
        frame_count = len(image_files)
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            raise IOError(f"Could not read the first image frame: {image_files[0]}")
        height, width = first_frame.shape[:2]
        fps = 30.0 # Default fallback FPS for image sequences
        
        for frame_idx, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"⚠️ Warning: Could not read frame image: {img_path}")
                continue
            yield frame, frame_idx, fps, frame_count, width, height
            
    else:
        # Standard video file input
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"OpenCV was unable to open the video file at: {video_path}")
            
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps is None:
                fps = 30.0
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame, frame_idx, fps, frame_count, width, height
                frame_idx += 1
        finally:
            cap.release()

def calculate_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (list or tuple): [x1, y1, x2, y2] bounding box coordinates.
        box2 (list or tuple): [x1, y1, x2, y2] bounding box coordinates.
        
    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0.0:
        return 0.0
        
    return intersection_area / union_area

class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=30):
        """
        A lightweight IoU-based tracker for matching human detections across frames.
        
        Args:
            iou_threshold (float): Minimum overlap required to match a detection to a track.
            max_lost_frames (int): Number of consecutive frames a track can be missing before deletion.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracked_persons = {}
        
    def update(self, detections, frame_idx):
        """
        Updates the tracks with new detections from the current frame.
        
        Args:
            detections (list): List of dicts, each with keys "bbox", "confidence", "keypoints".
            frame_idx (int): The current frame index.
            
        Returns:
            dict: Currently active tracks visible in this frame.
        """
        active_ids = list(self.tracked_persons.keys())
        matches = []
        
        # Calculate IoU between all current detections and existing tracked persons
        for det_idx, det in enumerate(detections):
            det_bbox = det["bbox"]
            for track_id in active_ids:
                track_bbox = self.tracked_persons[track_id]["bbox"]
                iou = calculate_iou(det_bbox, track_bbox)
                if iou >= self.iou_threshold:
                    matches.append((iou, det_idx, track_id))
                    
        # Sort matches by IoU in descending order (greedy matching)
        matches.sort(key=lambda x: x[0], reverse=True)
        
        matched_det_indices = set()
        matched_track_ids = set()
        
        for iou, det_idx, track_id in matches:
            if det_idx in matched_det_indices or track_id in matched_track_ids:
                continue
                
            matched_det_indices.add(det_idx)
            matched_track_ids.add(track_id)
            
            # Update tracked person details
            det = detections[det_idx]
            track_data = self.tracked_persons[track_id]
            track_data["bbox"] = det["bbox"]
            track_data["keypoints"] = det["keypoints"]
            track_data["confidence"] = det["confidence"]
            track_data["lost_frames"] = 0
            track_data["last_seen_frame"] = frame_idx
            
            centroid_x = (det["bbox"][0] + det["bbox"][2]) / 2.0
            centroid_y = (det["bbox"][1] + det["bbox"][3]) / 2.0
            track_data["centroid_history"].append((centroid_x, centroid_y, frame_idx))
            
            if len(track_data["centroid_history"]) > 100:
                track_data["centroid_history"].pop(0)
                
        # Register new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                centroid_x = (det["bbox"][0] + det["bbox"][2]) / 2.0
                centroid_y = (det["bbox"][1] + det["bbox"][3]) / 2.0
                
                self.tracked_persons[self.next_id] = {
                    "bbox": det["bbox"],
                    "keypoints": det["keypoints"],
                    "confidence": det["confidence"],
                    "centroid_history": [(centroid_x, centroid_y, frame_idx)],
                    "lost_frames": 0,
                    "is_fallen": False,
                    "last_fall_logged": False,
                    "fall_consecutive_frames": 0,
                    "last_seen_frame": frame_idx
                }
                self.next_id += 1
                
        # Handle lost tracks
        dead_tracks = []
        for track_id in active_ids:
            if track_id not in matched_track_ids:
                self.tracked_persons[track_id]["lost_frames"] += 1
                if self.tracked_persons[track_id]["lost_frames"] > self.max_lost_frames:
                    dead_tracks.append(track_id)
                    
        # Remove old lost tracks
        for track_id in dead_tracks:
            del self.tracked_persons[track_id]
            
        # Return tracks present in the current frame
        return {
            tid: data for tid, data in self.tracked_persons.items() 
            if data["last_seen_frame"] == frame_idx
        }

def draw_skeleton_and_box(frame, bbox, keypoints, person_id, is_fallen):
    """
    Overlays the bounding box, tracking ID, skeleton joints, and fall status on the frame.
    
    Args:
        frame (numpy.ndarray): OpenCV image frame.
        bbox (list): [x1, y1, x2, y2] coords.
        keypoints (numpy.ndarray): Pose keypoints of shape (17, 3) or (17, 2).
        person_id (int): Tracking ID assigned to the person.
        is_fallen (bool): Fall state indicator.
        
    Returns:
        numpy.ndarray: Annotated image frame.
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 0, 255) if is_fallen else (0, 255, 0)
    thickness = 3 if is_fallen else 2
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Label text
    status_str = "FALLEN" if is_fallen else "NORMAL"
    label = f"ID {person_id} [{status_str}]"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    
    # Text background box
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    num_cols = keypoints.shape[-1]
    
    # Draw connections
    for p1, p2 in SKELETON_CONNECTIONS:
        if p1 < len(keypoints) and p2 < len(keypoints):
            kp1, kp2 = keypoints[p1], keypoints[p2]
            conf1 = kp1[2] if num_cols == 3 else 1.0
            conf2 = kp2[2] if num_cols == 3 else 1.0
            
            if conf1 > 0.5 and conf2 > 0.5 and (kp1[0] != 0 or kp1[1] != 0) and (kp2[0] != 0 or kp2[1] != 0):
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(frame, pt1, pt2, (255, 128, 0), 2)
                
    # Draw joint nodes
    for idx, kp in enumerate(keypoints):
        conf = kp[2] if num_cols == 3 else 1.0
        if conf > 0.5 and (kp[0] != 0 or kp[1] != 0):
            pt = (int(kp[0]), int(kp[1]))
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)
            
    return frame

def calculate_torso_angle(keypoints):
    """
    Computes the torso vector's angle in degrees relative to the vertical axis.
    Torso vector is drawn from the hips midpoint to the shoulders midpoint.
    
    Args:
        keypoints (numpy.ndarray): Pose joints of shape (17, 3) or (17, 2).
        
    Returns:
        float or None: Angle in degrees, or None if keypoints are missing/low confidence.
    """
    num_cols = keypoints.shape[-1]
    required_kps = [5, 6, 11, 12] # Left/Right shoulders, Left/Right hips
    
    if max(required_kps) >= len(keypoints):
        return None
        
    # Check confidences and invalid coords
    for idx in required_kps:
        conf = keypoints[idx][2] if num_cols == 3 else 1.0
        if conf <= 0.4 or (keypoints[idx][0] == 0 and keypoints[idx][1] == 0):
            return None
            
    # Shoulders midpoint
    s_x = (keypoints[5][0] + keypoints[6][0]) / 2.0
    s_y = (keypoints[5][1] + keypoints[6][1]) / 2.0
    
    # Hips midpoint
    h_x = (keypoints[11][0] + keypoints[12][0]) / 2.0
    h_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
    
    # Torso vector components
    dx = s_x - h_x
    dy = s_y - h_y
    
    # Angle vs vertical axis (0 is vertical, 90 is horizontal)
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)

def calculate_aspect_ratio(bbox):
    """
    Computes the aspect ratio (width / height) of the bounding box.
    
    Args:
        bbox (list): [x1, y1, x2, y2] coords.
        
    Returns:
        float: Aspect ratio value.
    """
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(1e-6, y2 - y1)
    return width / height

def calculate_vertical_velocity(centroid_history, bbox_height, window_size=5):
    """
    Calculates vertical speed of the person, normalized by bbox height (scale-invariant).
    Positive values represent downward movement.
    
    Args:
        centroid_history (list): List of (x, y, frame_idx) tuples.
        bbox_height (float): Bounding box height in pixels.
        window_size (int): Temporal frame window size.
        
    Returns:
        float: Normalized vertical velocity.
    """
    if len(centroid_history) < 2:
        return 0.0
        
    curr_x, curr_y, curr_frame = centroid_history[-1]
    lookback_idx = max(0, len(centroid_history) - 1 - window_size)
    prev_x, prev_y, prev_frame = centroid_history[lookback_idx]
    
    frame_diff = curr_frame - prev_frame
    if frame_diff <= 0:
        return 0.0
        
    dy = curr_y - prev_y
    pix_velocity = dy / frame_diff
    
    # Normalize by bounding box height
    return pix_velocity / max(1e-6, bbox_height)

def log_incident(frame, frame_idx, person_id, confidence, video_name, evidence_dir, log_list):
    """
    Triggers an alert, saves the evidence frame, and logs the incident details.
    
    Args:
        frame (numpy.ndarray): Frame image where the fall was detected.
        frame_idx (int): The video frame index.
        person_id (int): Tracking ID of the person.
        confidence (float): YOLO detection confidence score.
        video_name (str): Filename of the processed video.
        evidence_dir (str): Folder path to save the JPEG snapshot.
        log_list (list): Reference to the list of log dicts to append to.
        
    Returns:
        str: Filename of the saved evidence snapshot.
    """
    # 1. Print visual console warning
    print(f"⚠️ FALL DETECTED - Person {person_id} at frame {frame_idx}")
    
    # 2. Format timestamp and filename
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence_filename = f"incident_{timestamp_str}_{person_id}.jpg"
    evidence_path = os.path.join(evidence_dir, evidence_filename)
    
    # 3. Save the snapshot frame (OpenCV BGR write)
    cv2.imwrite(evidence_path, frame)
    
    # 4. Append to list
    log_list.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "person_id": person_id,
        "confidence_score": float(confidence),
        "frame_number": frame_idx,
        "evidence_filename": evidence_filename,
        "video_name": video_name
    })
    
    return evidence_filename

def display_analytics_dashboard(logs_dir, save_plot=True):
    """
    Loads incident records from the CSV file and summary statistics,
    displays a formatted analytical dashboard, and optionally saves the plot.
    
    Args:
        logs_dir (str): Folder containing incident logs and statistics.
        save_plot (bool): If True, saves the plotted figures to the logs directory.
    """
    csv_path = os.path.join(logs_dir, "incident_log.csv")
    summary_path = os.path.join(logs_dir, "summary_stats.json")
    
    if not os.path.exists(csv_path):
        print("❌ No incident log CSV found. Process a video first.")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error loading incident log CSV: {e}")
        return
        
    print("\n" + "="*60)
    print("             FALL DETECTION MONITORING ANALYTICS")
    print("="*60)
    
    # Display printed stats
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                stats = json.load(f)
            print(f"Total Incident Alerts Triggered: {stats.get('total_incidents', 0)}")
            print(f"Total Unique People Tracked:      {stats.get('total_people_tracked', 0)}")
            print(f"Average Detection Confidence:    {stats.get('avg_confidence', 0.0):.2f}")
        except Exception as e:
            print(f"Warning: Could not read summary_stats.json: {e}")
    else:
        print(f"Total Incident Alerts Triggered: {len(df)}")
        if len(df) > 0:
            print(f"Total Unique People Fallen:      {df['person_id'].nunique()}")
            print(f"Average Detection Confidence:    {df['confidence_score'].mean():.2f}")
        else:
            print("Total Unique People Fallen:      0")
            print("Average Detection Confidence:    N/A")
            
    print("="*60 + "\n")
    
    if len(df) == 0:
        print("Zero incidents logged. Skip plotting.")
        return
        
    # Render Plots
    plt.ioff()  # Turn off interactive plotting to prevent popup blocking script
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Falls per Video/Session
    if 'video_name' in df.columns:
        video_counts = df['video_name'].value_counts()
        axes[0].bar(video_counts.index, video_counts.values, color='#4A90E2', edgecolor='black', zorder=2)
        axes[0].set_title("Incident Alert Count per Video Session", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Video File Name", fontsize=10)
        axes[0].set_ylabel("Number of Incidents", fontsize=10)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    else:
        axes[0].bar(["Session 1"], [len(df)], color='#4A90E2', edgecolor='black')
        axes[0].set_title("Incident Counts", fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)
        
    # Plot 2: Timeline Scatter Plot
    scatter = axes[1].scatter(
        df['frame_number'], 
        df['person_id'].astype(str), 
        s=df['confidence_score'] * 350, 
        c=df['confidence_score'], 
        cmap='plasma', 
        edgecolors='black', 
        alpha=0.85,
        zorder=2
    )
    axes[1].set_title("Incident Timeline (Frame vs. Person ID)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Frame Number", fontsize=10)
    axes[1].set_ylabel("Tracked Person ID", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label('YOLO Pose Confidence Score', fontsize=9)
    
    plt.tight_layout()
    if save_plot:
        plot_path = os.path.join(logs_dir, "analytics_dashboard.png")
        plt.savefig(plot_path, dpi=150)
        print(f"📈 Saved analytics dashboard plot to: {plot_path}")
    plt.close(fig)
