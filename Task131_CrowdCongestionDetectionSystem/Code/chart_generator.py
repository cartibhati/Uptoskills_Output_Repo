import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def generate_telemetry_charts(telemetry_csv_path, output_dir, video_name_only):
    """
    Reads the telemetry CSV log and generates a premium crowd density line chart over time.
    
    Args:
        telemetry_csv_path (str): Path to the telemetry CSV file.
        output_dir (str): Directory where the chart image will be saved.
        video_name_only (str): Name of the video (used in title and filename).
    """
    print(f"Generating crowd density telemetry chart for {video_name_only}...")
    
    if not os.path.exists(telemetry_csv_path):
        print(f"⚠️ Warning: Telemetry CSV file not found at: {telemetry_csv_path}. Skipping chart generation.")
        return
        
    try:
        # Load the CSV
        df = pd.read_csv(telemetry_csv_path)
        if df.empty:
            print("⚠️ Warning: Telemetry CSV is empty. Skipping chart generation.")
            return
            
        # Set styling parameters for a clean premium look
        plt.style.use('default')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        
        # Color palette for zones (Sleek modern hex codes)
        colors = {
            0: '#4f46e5',  # Indigo
            1: '#0d9488',  # Teal
            2: '#db2777',  # Pink
            3: '#d97706',  # Amber
            4: '#2563eb',  # Blue
        }
        
        # Get unique zones
        zones = df['zone'].unique()
        
        # Plot a line for each zone
        for idx, zone_name in enumerate(zones):
            zone_df = df[df['zone'] == zone_name].sort_values('frame_number')
            
            # Map x axis to frame numbers
            x = zone_df['frame_number'].values
            y = zone_df['crowd_count'].values
            
            # Check if there are congestion frames
            is_congested = zone_df['is_congested'].values
            
            color = colors.get(idx, '#6b7280') # fallback to gray
            
            # Plot the main trend line
            ax.plot(x, y, label=zone_name, color=color, linewidth=2, alpha=0.9)
            
            # Highlight congestion points with red dots / shaded regions
            congested_indices = [i for i, val in enumerate(is_congested) if val == 1]
            if congested_indices:
                ax.scatter(x[congested_indices], y[congested_indices], 
                           color='#ef4444', s=12, zorder=5, label='_nolegend_')
                
        # Format the X-axis: Map frame numbers to human-readable timestamps
        # We sample a set of frame numbers and corresponding timestamps to use as ticks
        # Get frame numbers and matching timestamps from the dataframe
        sample_df = df[['frame_number', 'timestamp']].drop_duplicates().sort_values('frame_number')
        
        if len(sample_df) > 1:
            total_frames = sample_df['frame_number'].max()
            # Decide tick spacing based on total frames (aim for 6-8 ticks)
            tick_step = max(1, total_frames // 7)
            
            tick_frames = list(range(0, int(total_frames) + 1, int(tick_step)))
            
            # Find the closest timestamp in our dataframe for each tick frame
            tick_labels = []
            for tf in tick_frames:
                closest_row = sample_df.iloc[(sample_df['frame_number'] - tf).abs().argsort()[:1]]
                if not closest_row.empty:
                    tick_labels.append(closest_row['timestamp'].values[0])
                else:
                    tick_labels.append(str(tf))
            
            ax.set_xticks(tick_frames)
            ax.set_xticklabels(tick_labels, rotation=0)
            
        # Customize labels and title
        formatted_title = f"Crowd Count Telemetry Trend - {video_name_only.replace('_', ' ').title()}"
        ax.set_title(formatted_title, fontsize=14, fontweight='bold', pad=15, color='#1f2937')
        ax.set_xlabel("Time (H:MM:SS)", fontsize=11, labelpad=8, color='#4b5563')
        ax.set_ylabel("Person Count", fontsize=11, labelpad=8, color='#4b5563')
        
        # Gridlines (subtle and modern)
        ax.grid(True, linestyle='--', alpha=0.5, color='#e5e7eb')
        ax.set_axisbelow(True)
        
        # Styling borders
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#d1d5db')
            
        # Legend configuration
        # Add a custom handle for congestion indicator
        handles, labels = ax.get_legend_handles_labels()
        
        # Check if there is any congestion in the entire run
        if (df['is_congested'] == 1).any():
            # Add a mock element for the legend representing active congestion
            congestion_handle = plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor='#ef4444', markersize=6, 
                                           label='Congestion Active')
            handles.append(congestion_handle)
            labels.append('Congestion Active')
            
        ax.legend(handles=handles, labels=labels, loc='upper left', frameon=True, 
                  facecolor='#ffffff', edgecolor='#e5e7eb', framealpha=0.9, shadow=False)
                  
        # Adjust layout and save
        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"{video_name_only}_crowd_density_chart.png")
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Telemetry chart saved successfully at: {chart_path}")
        
    except Exception as e:
        print(f"❌ Error generating telemetry chart: {e}")
        import traceback
        traceback.print_exc()
