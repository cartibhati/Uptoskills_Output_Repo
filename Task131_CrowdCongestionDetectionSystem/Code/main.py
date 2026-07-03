import os
import sys
import yaml
import time
import argparse
import datetime
import json
import cv2
import torch

# PyTorch 2.6+ compatibility patch to prevent weights_only security errors
try:
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass

# Add current directory to path to enable importing local modules in Code/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import video_frame_generator
from detector import PersonDetector
from tracker import SimpleIoUTracker
from zone_manager import ZoneManager
from alert_manager import AlertManager
from heatmap_generator import HeatmapGenerator
from output_writer import OutputWriter
from chart_generator import generate_telemetry_charts

def parse_args():
    parser = argparse.ArgumentParser(description="Task 131: Crowd Congestion Detection System")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml")),
        help="Path to the config.yaml configuration file."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Override path to the input video."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Override path to output directory."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load config file
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found at: {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Resolve overrides
    video_path = args.video if args.video else config.get("video_path", "")
    output_dir = args.output_dir if args.output_dir else config.get("output_dir", "Outputs")
    
    # Fully resolve absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(video_path):
        video_path = os.path.abspath(os.path.join(base_dir, video_path))
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(base_dir, output_dir))
        
    model_path = config.get("model_path", "Models/yolov8n.pt")
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(os.path.join(base_dir, model_path))
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Define specific output paths
    video_filename = os.path.basename(video_path)
    video_name_only, _ = os.path.splitext(video_filename)
    
    annotated_video_path = os.path.join(output_dir, f"{video_name_only}_annotated.mp4")
    telemetry_csv_path = os.path.join(output_dir, f"{video_name_only}_crowd_density.csv")
    alerts_log_path = os.path.join(output_dir, f"{video_name_only}_alerts.log")
    heatmap_final_path = os.path.join(output_dir, f"{video_name_only}_heatmap_final.png")
    
    print(f"\n⚡ Initializing Crowd Congestion Detection System (Task 131)")
    print(f" - Input Video:        {video_path}")
    print(f" - Output Directory:  {output_dir}")
    print(f" - YOLOv8 Model:      {model_path}")
    print(f" - Config File:       {args.config}")
    
    # 2. Instantiate core pipeline components
    print("\nLoading Person Detector...")
    detector = PersonDetector(
        model_path=model_path,
        confidence_threshold=config.get("detector", {}).get("confidence_threshold", 0.25)
    )
    print("✅ Detector initialized.")
    
    tracker_cfg = config.get("tracker", {})
    tracker = SimpleIoUTracker(
        iou_threshold=tracker_cfg.get("iou_threshold", 0.3),
        max_lost_frames=tracker_cfg.get("max_lost_frames", 30)
    )
    
    congestion_cfg = config.get("congestion", {})
    min_dur = congestion_cfg.get("min_duration_seconds", 5.0)
    zone_manager = ZoneManager(
        zone_config=config.get("zones", []),
        min_duration_seconds=min_dur
    )
    
    alert_manager = AlertManager(log_path=alerts_log_path)
    
    # Register alerts callback with zone manager
    zone_manager.register_callbacks(
        on_start=alert_manager.handle_congestion_start,
        on_end=alert_manager.handle_congestion_end
    )
    
    heatmap_cfg = config.get("heatmap", {})
    heatmap_gen = HeatmapGenerator(
        output_dir=output_dir,
        colormap=heatmap_cfg.get("colormap", "COLORMAP_JET"),
        alpha=heatmap_cfg.get("alpha", 0.6)
    )
    
    # 3. Processing loop
    start_time = time.time()
    processed_frames = 0
    writer = None
    
    try:
        frame_generator = video_frame_generator(video_path)
        snapshot_interval = heatmap_cfg.get("snapshot_interval_seconds", 5.0)
        
        for frame, frame_idx, fps, frame_count, width, height in frame_generator:
            # Resize frame to 768x432 for robust spatial zone boundaries (config is calibrated for 768x432)
            target_width, target_height = 768, 432
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height))
                width, height = target_width, target_height

            # Initialize OutputWriter on first frame
            if writer is None:
                writer = OutputWriter(
                    video_output_path=annotated_video_path,
                    csv_output_path=telemetry_csv_path,
                    fps=fps,
                    width=width,
                    height=height
                )
                print(f"Video Info: {width}x{height} pixels | {fps:.2f} FPS | {frame_count} total frames")
                print("Processing video frames...")
                
            # A. Run YOLO person detection
            detections = detector.detect(frame)
            
            # B. Track people
            active_tracks = tracker.update(detections, frame_idx)
            
            # C. Update Zone Manager (assigns tracks to zones, computes density & triggers state machine)
            zone_states = zone_manager.update(active_tracks, frame_idx, fps)
            
            # D. Annotate frame
            annotated_frame = writer.annotate_frame(frame, active_tracks, zone_states, config.get("zones", []))
            
            # E. Write output video frame and telemetry CSV
            writer.write_frame(annotated_frame)
            
            # Generate human-readable time elapsed in video for CSV
            elapsed_sec = frame_idx / fps
            td = datetime.timedelta(seconds=elapsed_sec)
            video_time_str = str(td).split('.')[0] # format: H:MM:SS
            writer.write_telemetry(video_time_str, frame_idx, zone_states)
            
            # F. Update density heatmap
            heatmap_gen.update(active_tracks, frame)
            
            # Save periodic heatmap snapshots
            if snapshot_interval > 0 and frame_idx > 0 and int(frame_idx) % int(fps * snapshot_interval) == 0:
                snap_num = int(frame_idx // (fps * snapshot_interval))
                snapshot_name = f"{video_name_only}_heatmap_snapshot_{snap_num:03d}.png"
                heatmap_gen.save_snapshot(snapshot_name, frame)
                
            processed_frames += 1
            if processed_frames % 50 == 0 or processed_frames == frame_count:
                progress = (processed_frames / frame_count) * 100
                print(f" ⏳ Progress: {processed_frames}/{frame_count} frames ({progress:.1f}%)")
                
        # 4. Cleanup and Save final results
        if writer is not None:
            writer.close()
            
        if processed_frames == 0:
            print("❌ Error: No frames were loaded. Check video file.")
            sys.exit(1)
            
        # Save final heatmap
        heatmap_gen.save_snapshot(f"{video_name_only}_heatmap_final.png")
        
        # Save summary statistics JSON
        elapsed_time = time.time() - start_time
        avg_fps = processed_frames / elapsed_time
        
        summary = {
            "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames_processed": processed_frames,
            "total_time_seconds": float(round(elapsed_time, 2)),
            "average_fps": float(round(avg_fps, 2)),
            "total_congestion_events": len(zone_manager.congestion_history),
            "video_fps": float(fps),
            "video_resolution": f"{width}x{height}",
            "congestion_history": zone_manager.congestion_history
        }
        
        summary_path = os.path.join(output_dir, f"{video_name_only}_summary_stats.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Generate telemetry trend charts
        generate_telemetry_charts(telemetry_csv_path, output_dir, video_name_only)
            
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total Processing Time: {elapsed_time:.2f} seconds")
        print(f"Average Processing Speed: {avg_fps:.2f} FPS")
        print(f"Annotated Video:       {annotated_video_path}")
        print(f"Telemetry CSV Log:     {telemetry_csv_path}")
        print(f"Alerts History Log:    {alerts_log_path}")
        print(f"Final Heatmap Image:   {heatmap_final_path}")
        print(f"Summary Stats JSON:    {summary_path}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Critical pipeline failure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
