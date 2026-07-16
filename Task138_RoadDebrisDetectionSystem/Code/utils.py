import os
import cv2
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def video_frame_generator(video_path):
    """
    Opens a video file OR a directory containing sequential images, and yields frames 
    sequentially along with sequence metadata.
    
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
        
    # Check if the path is a directory of images
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
                print(f"[WARN] Warning: Could not read frame image: {img_path}")
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
        A lightweight class-aware IoU-based tracker for matching detections across frames.
        
        Args:
            iou_threshold (float): Minimum overlap required to match a detection to a track.
            max_lost_frames (int): Number of consecutive frames a track can be missing before deletion.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracked_objects = {}
        
    def update(self, detections, frame_idx):
        """
        Updates the tracks with new detections from the current frame.
        
        Args:
            detections (list): List of dicts, each with keys "bbox", "confidence", "class_id", "class_name".
            frame_idx (int): The current frame index.
            
        Returns:
            dict: Currently active tracks visible in this frame.
        """
        active_ids = list(self.tracked_objects.keys())
        matches = []
        
        # Calculate IoU between all current detections and existing tracked objects of the same class
        for det_idx, det in enumerate(detections):
            det_bbox = det["bbox"]
            det_cls = det["class_id"]
            for track_id in active_ids:
                track_data = self.tracked_objects[track_id]
                # Class-aware matching: only match objects of the same class
                if track_data["class_id"] == det_cls:
                    track_bbox = track_data["bbox"]
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
            
            # Update tracked object details
            det = detections[det_idx]
            track_data = self.tracked_objects[track_id]
            track_data["bbox"] = det["bbox"]
            track_data["confidence"] = det["confidence"]
            track_data["lost_frames"] = 0
            track_data["last_seen_frame"] = frame_idx
            
            centroid_x = (det["bbox"][0] + det["bbox"][2]) / 2.0
            centroid_y = (det["bbox"][1] + det["bbox"][3]) / 2.0
            track_data["centroid_history"].append((centroid_x, centroid_y, frame_idx))
            
            if len(track_data["centroid_history"]) > 150: # Keep a longer history for velocity calculations
                track_data["centroid_history"].pop(0)
                
        # Register new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                centroid_x = (det["bbox"][0] + det["bbox"][2]) / 2.0
                centroid_y = (det["bbox"][1] + det["bbox"][3]) / 2.0
                
                self.tracked_objects[self.next_id] = {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "centroid_history": [(centroid_x, centroid_y, frame_idx)],
                    "lost_frames": 0,
                    "first_seen_frame": frame_idx,
                    "last_seen_frame": frame_idx,
                    "stationary_frames": 0,
                    "is_hazard": False,
                    "hazard_logged": False
                }
                self.next_id += 1
                
        # Handle lost tracks
        dead_tracks = []
        for track_id in active_ids:
            if track_id not in matched_track_ids:
                self.tracked_objects[track_id]["lost_frames"] += 1
                if self.tracked_objects[track_id]["lost_frames"] > self.max_lost_frames:
                    dead_tracks.append(track_id)
                    
        # Remove old lost tracks
        for track_id in dead_tracks:
            del self.tracked_objects[track_id]
            
        # Return tracks present in the current frame
        return {
            tid: data for tid, data in self.tracked_objects.items() 
            if data["last_seen_frame"] == frame_idx
        }

def is_inside_polygon(point, polygon):
    """
    Checks if a point (x, y) is inside a polygon using OpenCV's pointPolygonTest.
    
    Args:
        point (tuple): (x, y) coordinates of the point.
        polygon (list): List of (x, y) vertices defining the polygon.
        
    Returns:
        bool: True if the point is inside or on the edge of the polygon, False otherwise.
    """
    poly_arr = np.array(polygon, dtype=np.float32)
    pt = (float(point[0]), float(point[1]))
    # cv2.pointPolygonTest returns positive value if inside, 0 if on edge, negative if outside
    result = cv2.pointPolygonTest(poly_arr, pt, False)
    return result >= 0

def log_debris_incident(frame, frame_idx, track_id, track_data, video_name, evidence_dir, log_list):
    """
    Logs a road obstacle/debris incident, prints an alert, and saves an evidence frame.
    
    Args:
        frame (numpy.ndarray): The current video frame.
        frame_idx (int): The current frame index.
        track_id (int): Persistent ID of the tracked object.
        track_data (dict): The state dictionary of the track.
        video_name (str): Filename of the input video.
        evidence_dir (str): Folder path to save the JPEG snapshot.
        log_list (list): Reference to the global log list to append to.
        
    Returns:
        str: Filename of the saved evidence snapshot.
    """
    cls_name = track_data["class_name"]
    conf = track_data["confidence"]
    
    print(f"[ALERT] STATIONARY ROAD OBSTACLE: {cls_name.upper()} (ID {track_id}) detected at frame {frame_idx}")
    
    # Format timestamp and filename
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence_filename = f"incident_{timestamp_str}_{cls_name}_id_{track_id}.jpg"
    evidence_path = os.path.join(evidence_dir, evidence_filename)
    
    # Save frame
    cv2.imwrite(evidence_path, frame)
    
    # Append structured log
    log_list.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_id": f"{os.path.splitext(video_name)[0]}_id_{track_id}",
        "object_id": track_id,
        "class_name": cls_name,
        "confidence_score": float(conf),
        "frame_number": frame_idx,
        "evidence_filename": evidence_filename,
        "video_name": video_name,
        "status": "STATIONARY_OBSTACLE"
    })
    
    return evidence_filename

def display_analytics_dashboard(logs_dir, save_plot=True):
    """
    Loads incident records from the CSV file and displays/saves a formatted analytics dashboard.
    
    Args:
        logs_dir (str): Folder containing incident logs.
        save_plot (bool): Whether to save the plotted dashboard figure as a PNG.
    """
    csv_path = os.path.join(logs_dir, "incident_log.csv")
    summary_path = os.path.join(logs_dir, "summary_stats.json")
    
    if not os.path.exists(csv_path):
        print("[ERROR] No incident log CSV found. Process a video first.")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Error loading incident log CSV: {e}")
        return
        
    print("\n" + "="*60)
    print("             ROAD DEBRIS DETECTION SYSTEM ANALYTICS")
    print("="*60)
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                stats = json.load(f)
            print(f"Total Incidents Logged:        {stats.get('total_incidents', 0)}")
            print(f"Unique Obstacles Tracked:      {stats.get('unique_obstacles_tracked', 0)}")
            print(f"Average Detection Confidence:  {stats.get('avg_confidence', 0.0):.2f}")
            print(f"Most Frequent Hazard Class:    {stats.get('most_frequent_hazard_class', 'N/A')}")
        except Exception as e:
            print(f"Warning: Could not read summary_stats.json: {e}")
    else:
        print(f"Total Incidents Logged:        {len(df)}")
        if len(df) > 0:
            print(f"Unique Obstacles Tracked:      {df['object_id'].nunique()}")
            print(f"Average Detection Confidence:  {df['confidence_score'].mean():.2f}")
            print(f"Most Frequent Hazard Class:    {df['class_name'].mode()[0] if not df['class_name'].empty else 'N/A'}")
            
    print("="*60 + "\n")
    
    if len(df) == 0:
        print("Zero incidents logged. Skipping dashboard plot rendering.")
        return
        
    plt.ioff()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color Palette: Harmonious slate blue and warm coral
    primary_color = '#E06666' # Soft Red/Coral
    secondary_color = '#4A90E2' # Sleek Blue
    
    # Plot 1: Incident Count by Object Class
    class_counts = df['class_name'].value_counts()
    axes[0].bar(class_counts.index, class_counts.values, color=secondary_color, edgecolor='black', zorder=2)
    axes[0].set_title("Incident Alert Count by Obstacle/Debris Class", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Object Class", fontsize=10)
    axes[0].set_ylabel("Number of Incidents", fontsize=10)
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    
    # Plot 2: Timeline Scatter Plot
    # X: frame_number, Y: class_name, Size/Color: confidence_score
    scatter = axes[1].scatter(
        df['frame_number'], 
        df['class_name'].astype(str) + " (ID " + df['object_id'].astype(str) + ")",
        s=df['confidence_score'] * 350, 
        c=df['confidence_score'], 
        cmap='plasma', 
        edgecolors='black', 
        alpha=0.85,
        zorder=2
    )
    axes[1].set_title("Incident Timeline (Frame vs. Obstacle ID)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Frame Number", fontsize=10)
    axes[1].set_ylabel("Hazard Type & ID", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label('YOLOv8 Detection Confidence Score', fontsize=9)
    
    plt.tight_layout()
    if save_plot:
        plot_path = os.path.join(logs_dir, "analytics_dashboard.png")
        plt.savefig(plot_path, dpi=150)
        print(f"[INFO] Saved analytics dashboard plot to: {plot_path}")
    plt.close(fig)
