import os
import sys
import cv2
import argparse
import time
import json
import torch
import pandas as pd
import numpy as np

# PyTorch 2.6 compatibility patch for loading YOLO models safely
try:
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass

from ultralytics import YOLO

# Add the directory containing utils to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    video_frame_generator,
    SimpleIoUTracker,
    is_inside_polygon,
    log_debris_incident,
    display_analytics_dashboard
)

def parse_args():
    parser = argparse.ArgumentParser(description="AI-Based Road Debris & Obstacle Detection System")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file or image sequence directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save the annotated output video. If empty, auto-saves to Outputs/."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Models", "yolov8n.pt")),
        help="Path to the YOLOv8 model weight file."
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="0.05,0.95,0.40,0.50,0.60,0.50,0.95,0.95",
        help="Normalized trapezoid road ROI polygon vertices (x1,y1,x2,y2,x3,y3,x4,y4) between 0.0 and 1.0."
    )
    parser.add_argument(
        "--debris_classes",
        type=str,
        default="backpack,umbrella,handbag,suitcase,bottle,cup,chair,couch,potted plant,sports ball,frisbee,stop sign,book,box",
        help="Comma-separated list of COCO object classes representing immediate debris candidates."
    )
    parser.add_argument(
        "--vehicle_classes",
        type=str,
        default="person,car,motorcycle,bus,truck,bicycle,dog,cat,horse,cow,sheep",
        help="Comma-separated list of COCO object classes representing vehicles/people (slow warning time)."
    )
    parser.add_argument(
        "--debris_persistence",
        type=float,
        default=1.5,
        help="Stationary time in seconds required to trigger an alert for road debris."
    )
    parser.add_argument(
        "--vehicle_persistence",
        type=float,
        default=4.0,
        help="Stationary time in seconds required to trigger an alert for vehicles/people."
    )
    parser.add_argument(
        "--stationary_threshold",
        type=float,
        default=0.015,
        help="Centroid displacement threshold (as fraction of image width) to qualify as stationary."
    )
    parser.add_argument(
        "--camera_motion",
        type=str,
        choices=["stationary", "moving"],
        default="stationary",
        help="Camera setup: stationary (surveillance) or moving (dashcam/moving vehicle)."
    )
    parser.add_argument(
        "--moving_persistence",
        type=float,
        default=0.2,
        help="Alert persistence duration in seconds for moving camera mode."
    )
    parser.add_argument(
        "--speed_threshold",
        type=float,
        default=0.002,
        help="Vertical downward speed threshold (fraction of height/frame) to qualify as approaching."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup base output directories relative to script
    code_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.abspath(os.path.join(code_dir, "..", "Outputs"))
    evidence_dir = os.path.join(outputs_dir, "evidence_frames")
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)
    
    # Resolve input path
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"[ERROR] Input path does not exist: {video_path}")
        sys.exit(1)
        
    video_filename = os.path.basename(video_path.rstrip('/\\')) or "image_sequence"
    video_name_only, _ = os.path.splitext(video_filename)
    
    # Resolve output video path
    if args.output:
        output_video_path = args.output
    else:
        output_video_path = os.path.join(outputs_dir, f"{video_name_only}_annotated.mp4")
        
    # Parse class lists
    debris_class_list = [c.strip().lower() for c in args.debris_classes.split(',') if c.strip()]
    vehicle_class_list = [c.strip().lower() for c in args.vehicle_classes.split(',') if c.strip()]
    all_allowed_classes = set(debris_class_list + vehicle_class_list)
    
    # Parse Road ROI coordinates
    try:
        roi_floats = [float(x) for x in args.roi.split(',') if x.strip()]
        if len(roi_floats) % 2 != 0 or len(roi_floats) < 6:
            raise ValueError("ROI must have at least 3 vertices (6 coordinates) and be even-numbered.")
        # Group into pairs of (x, y)
        normalized_roi = [(roi_floats[i], roi_floats[i+1]) for i in range(0, len(roi_floats), 2)]
    except Exception as e:
        print(f"[ERROR] Error parsing ROI polygon: {e}")
        print("Falling back to default road ROI.")
        normalized_roi = [(0.05, 0.95), (0.40, 0.50), (0.60, 0.50), (0.95, 0.95)]
        
    print(f"\n[INFO] Initializing Road Debris Detection Pipeline")
    print(f" - Input Path:             {video_path}")
    print(f" - Output Video Path:      {output_video_path}")
    print(f" - YOLOv8 Model:           {args.model}")
    print(f" - Debris Persistence:     {args.debris_persistence}s")
    print(f" - Vehicle Persistence:    {args.vehicle_persistence}s")
    print(f" - Displacement Threshold: {args.stationary_threshold * 100:.2f}% of width")
    print(f" - Road ROI vertices:     {normalized_roi}")
    
    if not os.path.exists(args.model):
        print(f"[ERROR] Model weights file not found at: {args.model}")
        print("Please ensure yolov8n.pt is downloaded and located in the Models/ directory.")
        sys.exit(1)
        
    print("\nLoading YOLOv8 model...")
    model = YOLO(args.model)
    print("[OK] Model loaded successfully.\n")
    
    start_time = time.time()
    processed_frames = 0
    total_tracked_hazards = set()
    incident_logs = []
    
    tracker = SimpleIoUTracker(iou_threshold=0.35, max_lost_frames=30)
    
    try:
        frame_generator = video_frame_generator(video_path)
        out_writer = None
        
        for frame, frame_idx, fps, frame_count, width, height in frame_generator:
            if out_writer is None:
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                print(f"Video specs: {width}x{height} pixels | {fps:.2f} FPS | {frame_count} total frames")
                print("Processing frames...")
                
                # Scale normalized ROI to actual pixel sizes
                pixel_roi = [(int(pt[0] * width), int(pt[1] * height)) for pt in normalized_roi]
                pixel_roi_np = np.array(pixel_roi, dtype=np.int32)
                
            # Run YOLOv8 detection
            results = model.predict(frame, verbose=False)
            result = results[0]
            
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                names = result.names
                
                for idx in range(len(boxes)):
                    class_name = names[int(classes[idx])].lower()
                    # Map generic 'bottle' or 'cup' or similar to debris if needed
                    # If YOLOv8 detects anything that maps to the allowed list, include it
                    if class_name in all_allowed_classes and confs[idx] >= 0.3:
                        detections.append({
                            "bbox": list(boxes[idx]),
                            "confidence": float(confs[idx]),
                            "class_id": int(classes[idx]),
                            "class_name": class_name
                        })
                        
            # Update IoU tracker
            current_tracks = tracker.update(detections, frame_idx)
            
            # Process each active track in this frame
            annotated_frame = frame.copy()
            
            # Draw Road Region of Interest polygon overlay
            # Draw semi-transparent fill for the ROI
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [pixel_roi_np], (255, 100, 0)) # Blue-ish fill
            cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
            # Draw outline and label
            cv2.polylines(annotated_frame, [pixel_roi_np], True, (255, 120, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(annotated_frame, "ROAD ROI (HAZARD DETECTION ZONE)", (pixel_roi[1][0] - 20, pixel_roi[1][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2, lineType=cv2.LINE_AA)
            
            active_hazards_in_frame = []
            
            for track_id, track in current_tracks.items():
                history = track["centroid_history"]
                curr_cx, curr_cy, _ = history[-1]
                
                # Check if centroid is inside the Road ROI
                in_roi = is_inside_polygon((curr_cx, curr_cy), pixel_roi)
                
                # Determine stationary or approaching hazard state
                is_stationary = False
                is_approaching = False
                
                if in_roi and len(history) >= 5:
                    if args.camera_motion == "moving":
                        # In moving camera, stationary road objects move downwards rapidly in the frame.
                        # Check if vertical velocity is positive and significant (moving closer).
                        window_history = history[-5:]
                        curr_cx, curr_cy, curr_f = window_history[-1]
                        prev_cx, prev_cy, prev_f = window_history[0]
                        fdiff = curr_f - prev_f
                        if fdiff > 0:
                            vy = (curr_cy - prev_cy) / fdiff
                            pixel_speed_thresh = args.speed_threshold * height
                            if vy >= pixel_speed_thresh:
                                is_approaching = True
                    else:
                        # Stationary camera mode: displacement must be very low
                        rolling_window = max(5, int(fps * 1.5))
                        window_history = history[-rolling_window:]
                        displacements = [
                            np.sqrt((curr_cx - h_cx)**2 + (curr_cy - h_cy)**2)
                            for h_cx, h_cy, _ in window_history
                        ]
                        max_displacement = max(displacements)
                        pixel_threshold = args.stationary_threshold * width
                        if max_displacement <= pixel_threshold:
                            is_stationary = True
                            
                # Update stationary/approaching frame count
                if in_roi and (is_stationary or is_approaching):
                    track["stationary_frames"] += 1
                else:
                    track["stationary_frames"] = 0
                    track["is_hazard"] = False
                    
                # Evaluate persistence logic
                stationary_duration = track["stationary_frames"] / fps
                cls_name = track["class_name"]
                
                if args.camera_motion == "moving":
                    # In moving mode, debris is immediately a hazard, vehicles require moving_persistence
                    required_persistence = 0.1 if cls_name in debris_class_list else args.moving_persistence
                else:
                    required_persistence = args.debris_persistence if cls_name in debris_class_list else args.vehicle_persistence
                
                if track["stationary_frames"] > 0 and stationary_duration >= required_persistence:
                    was_hazard = track["is_hazard"]
                    track["is_hazard"] = True
                    total_tracked_hazards.add(track_id)
                    active_hazards_in_frame.append((track_id, cls_name))
                    
                    # Log incident on first transition
                    if not was_hazard and not track["hazard_logged"]:
                        log_debris_incident(
                            frame=frame,
                            frame_idx=frame_idx,
                            track_id=track_id,
                            track_data=track,
                            video_name=video_filename,
                            evidence_dir=evidence_dir,
                            log_list=incident_logs
                        )
                        track["hazard_logged"] = True
                        
                # Draw visual annotations
                x1, y1, x2, y2 = map(int, track["bbox"])
                
                # Bounding box color coding:
                # - Red: confirmed road hazard
                # - Yellow: stationary warning (stopped in road, but hasn't reached persistence yet)
                # - Green: normal moving object
                if track["is_hazard"]:
                    box_color = (0, 0, 255) # Red
                    box_thick = 3
                    status_lbl = "HAZARD: OBSTACLE"
                elif track["stationary_frames"] > 0:
                    box_color = (0, 255, 255) # Yellow
                    box_thick = 2
                    status_lbl = f"STOPPED ({stationary_duration:.1f}s)"
                else:
                    box_color = (0, 255, 0) # Green
                    box_thick = 2
                    status_lbl = "MOVING"
                    
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, box_thick)
                
                # Overlay label text
                label = f"ID {track_id} | {cls_name} | {status_lbl}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 18), (x1 + w, y1), box_color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                
                # Draw fading centroid history track (premium micro-animation)
                for h_idx in range(1, len(history)):
                    pt1 = (int(history[h_idx - 1][0]), int(history[h_idx - 1][1]))
                    pt2 = (int(history[h_idx][0]), int(history[h_idx][1]))
                    # Fading color based on age
                    age_alpha = h_idx / len(history)
                    line_color = (
                        int(box_color[0] * age_alpha + 128 * (1 - age_alpha)),
                        int(box_color[1] * age_alpha + 128 * (1 - age_alpha)),
                        int(box_color[2] * age_alpha)
                    )
                    cv2.line(annotated_frame, pt1, pt2, line_color, 2, lineType=cv2.LINE_AA)
                    
            # Draw header dashboard warning if hazards are present in the current frame
            if active_hazards_in_frame:
                cv2.rectangle(annotated_frame, (0, 0), (width, 40), (0, 0, 180), -1)
                hazards_desc = ", ".join([f"{name} (ID {tid})" for tid, name in active_hazards_in_frame[:3]])
                if len(active_hazards_in_frame) > 3:
                    hazards_desc += "..."
                warning_msg = f"WARNING: ROAD BLOCKED - Active Obstacles: {hazards_desc}"
                cv2.putText(annotated_frame, warning_msg, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                
            out_writer.write(annotated_frame)
            processed_frames += 1
            
            if processed_frames % 50 == 0 or processed_frames == frame_count:
                progress = (processed_frames / frame_count) * 100
                print(f" [INFO] Processed {processed_frames}/{frame_count} frames ({progress:.1f}%)")
                
        # Close output stream
        if out_writer is not None:
            out_writer.release()
            
        if processed_frames == 0:
            print("[ERROR] Processing finished, but no frames were loaded. Check video file.")
            return
            
        # Export CSV log
        log_cols = ["timestamp", "event_id", "object_id", "class_name", "confidence_score", "frame_number", "evidence_filename", "video_name", "status"]
        log_df = pd.DataFrame(incident_logs, columns=log_cols)
        csv_path = os.path.join(outputs_dir, "incident_log.csv")
        
        # Merge with existing logs to preserve history across video runs
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                existing_df = pd.read_csv(csv_path)
                log_df = pd.concat([existing_df, log_df], ignore_index=True)
                log_df = log_df.drop_duplicates(subset=["timestamp", "object_id", "frame_number", "video_name"])
            except Exception:
                pass
                
        log_df.to_csv(csv_path, index=False)
        print(f"[INFO] Incident log CSV written/updated: {csv_path}")
        
        # Save summary stats JSON
        avg_conf = log_df['confidence_score'].mean() if len(log_df) > 0 else 0.0
        most_freq_class = log_df['class_name'].mode()[0] if len(log_df) > 0 and not log_df['class_name'].empty else "N/A"
        summary_stats = {
            "total_incidents": len(log_df),
            "unique_obstacles_tracked": len(total_tracked_hazards),
            "avg_confidence": float(avg_conf),
            "most_frequent_hazard_class": most_freq_class
        }
        summary_path = os.path.join(outputs_dir, "summary_stats.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=4)
            
        # Timing Stats
        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed
        
        print("\n" + "="*60)
        print("[SUCCESS] ROAD DEBRIS DETECTION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total Processing Time: {elapsed:.2f} seconds")
        print(f"Average FPS Achieved:  {avg_fps:.2f} FPS")
        print(f"Annotated Video Saved: {output_video_path}")
        print("="*60 + "\n")
        
        # Save analytics dashboard
        display_analytics_dashboard(outputs_dir, save_plot=True)
        
    except Exception as e:
        print(f"[ERROR] Critical pipeline failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
