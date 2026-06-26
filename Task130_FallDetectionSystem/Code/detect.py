import os
import sys
import cv2
import argparse
import time
import json
import torch
import pandas as pd
from ultralytics import YOLO

# Add the directory containing utils to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    video_frame_generator,
    SimpleIoUTracker,
    calculate_torso_angle,
    calculate_aspect_ratio,
    draw_skeleton_and_box,
    log_incident,
    display_analytics_dashboard
)

# PyTorch 2.6 compatibility patch
try:
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="AI-Based Fall Detection System")
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
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Models", "yolov8n-pose.pt")),
        help="Path to the YOLOv8-pose model weight file."
    )
    parser.add_argument(
        "--threshold_angle",
        type=float,
        default=60.0,
        help="Torso angle threshold vs vertical in degrees."
    )
    parser.add_argument(
        "--threshold_aspect",
        type=float,
        default=0.6,
        help="Bounding box aspect ratio (width/height) threshold."
    )
    parser.add_argument(
        "--persistence",
        type=float,
        default=1.0,
        help="Fall state persistence duration in seconds."
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
    
    # Resolve input video name
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"❌ Error: Input path does not exist: {video_path}")
        sys.exit(1)
        
    video_filename = os.path.basename(video_path.rstrip('/\\')) or "image_sequence"
    video_name_only, _ = os.path.splitext(video_filename)
    
    # Resolve output video path
    if args.output:
        output_video_path = args.output
    else:
        output_video_path = os.path.join(outputs_dir, f"{video_name_only}_annotated.mp4")
        
    print(f"\n🎬 Initializing Fall Detection Pipeline")
    print(f" - Input Path:        {video_path}")
    print(f" - Output Video Path: {output_video_path}")
    print(f" - YOLOv8 Model:      {args.model}")
    print(f" - Angle Threshold:   {args.threshold_angle}°")
    print(f" - Aspect Threshold:  {args.threshold_aspect}")
    print(f" - Persistence Secs:  {args.persistence}s")
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model weights file not found at: {args.model}")
        print("Please ensure yolov8n-pose.pt is downloaded and located in the Models/ directory.")
        sys.exit(1)
        
    print("\nLoading YOLOv8 Pose model...")
    model = YOLO(args.model)
    print("✅ Model loaded successfully.\n")
    
    start_time = time.time()
    processed_frames = 0
    total_tracked_people = set()
    incident_logs = []
    
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=30)
    
    try:
        frame_generator = video_frame_generator(video_path)
        out_writer = None
        
        for frame, frame_idx, fps, frame_count, width, height in frame_generator:
            if out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                print(f"Video specs: {width}x{height} pixels | {fps:.2f} FPS | {frame_count} total frames")
                print("Processing frames...")
                
            # Run YOLO pose detection
            results = model.predict(frame, verbose=False)
            result = results[0]
            
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                kps = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
                
                for idx in range(len(boxes)):
                    # COCO class 0 is Person
                    if int(classes[idx]) == 0:
                        detections.append({
                            "bbox": list(boxes[idx]),
                            "confidence": float(confs[idx]),
                            "keypoints": kps[idx] if kps is not None else None
                        })
                        
            # Update IoU tracker
            current_tracks = tracker.update(detections, frame_idx)
            
            # Record unique tracked IDs
            for tid in current_tracks.keys():
                total_tracked_people.add(tid)
                
            # Process detections per track
            annotated_frame = frame.copy()
            for track_id, track in current_tracks.items():
                if track["keypoints"] is None:
                    continue
                    
                # Compute heuristics
                torso_angle = calculate_torso_angle(track["keypoints"])
                bbox = track["bbox"]
                aspect_ratio = calculate_aspect_ratio(bbox)
                
                # Check fall criteria
                is_instantly_falling = (
                    torso_angle is not None 
                    and torso_angle >= args.threshold_angle 
                    and aspect_ratio >= args.threshold_aspect
                )
                
                if is_instantly_falling:
                    track["fall_consecutive_frames"] += 1
                    required_frames = max(1, int(fps * args.persistence))
                    
                    if track["fall_consecutive_frames"] >= required_frames:
                        was_fallen = track["is_fallen"]
                        track["is_fallen"] = True
                        
                        # Save evidence frame and log incident on transition to fallen state
                        if not was_fallen and not track["last_fall_logged"]:
                            log_incident(
                                frame=frame,
                                frame_idx=frame_idx,
                                person_id=track_id,
                                confidence=track["confidence"],
                                video_name=video_filename,
                                evidence_dir=evidence_dir,
                                log_list=incident_logs
                            )
                            track["last_fall_logged"] = True
                else:
                    # Reset tracker state if person recovers / is not falling
                    track["fall_consecutive_frames"] = 0
                    track["is_fallen"] = False
                    track["last_fall_logged"] = False
                    
                # Draw skeleton annotations on the frame
                annotated_frame = draw_skeleton_and_box(
                    annotated_frame, 
                    track["bbox"], 
                    track["keypoints"], 
                    track_id, 
                    track["is_fallen"]
                )
                
            out_writer.write(annotated_frame)
            processed_frames += 1
            
            if processed_frames % 50 == 0 or processed_frames == frame_count:
                progress = (processed_frames / frame_count) * 100
                print(f" ⏳ Processed {processed_frames}/{frame_count} frames ({progress:.1f}%)")
                
        # Close output stream
        if out_writer is not None:
            out_writer.release()
            
        if processed_frames == 0:
            print("❌ Processing finished, but no frames were loaded. Check video file.")
            return
            
        # Export CSV log
        log_df = pd.DataFrame(incident_logs)
        csv_path = os.path.join(outputs_dir, "incident_log.csv")
        
        # Merge with existing log to preserve history across video runs
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                existing_df = pd.read_csv(csv_path)
                log_df = pd.concat([existing_df, log_df], ignore_index=True)
                log_df = log_df.drop_duplicates(subset=["timestamp", "person_id", "frame_number", "video_name"])
            except Exception:
                pass
                
        log_df.to_csv(csv_path, index=False)
        print(f"📊 Incident log CSV written/updated: {csv_path}")
        
        # Save summary stats JSON
        avg_conf = log_df['confidence_score'].mean() if len(log_df) > 0 else 0.0
        summary_stats = {
            "total_incidents": len(log_df),
            "avg_confidence": float(avg_conf),
            "total_people_tracked": len(total_tracked_people)
        }
        summary_path = os.path.join(outputs_dir, "summary_stats.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=4)
            
        # Timing Stats
        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed
        
        print("\n" + "="*60)
        print("🎉 PIPELINE RUN COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total Processing Time: {elapsed:.2f} seconds")
        print(f"Average FPS Achieved:  {avg_fps:.2f} FPS")
        print(f"Annotated Video Saved: {output_video_path}")
        print("="*60 + "\n")
        
        # Trigger analytics dashboard rendering
        display_analytics_dashboard(outputs_dir, save_plot=True)
        
    except Exception as e:
        print(f"❌ Critical pipeline failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
