import json
import google.generativeai as genai
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

ANALYSIS_DIR = "traffic_analysis"

def load_traffic_data(json_path="vis_preds/analysis_data.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Analysis data not found at {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_visualizations(data, output_dir=ANALYSIS_DIR):
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Output directory path: {os.path.abspath(output_dir)}")
    
    plt.figure(figsize=(12, 6))
    incoming_speeds = [event['speed_kph'] for event in data['speed_events'] if event['direction'] == 'incoming']
    outgoing_speeds = [event['speed_kph'] for event in data['speed_events'] if event['direction'] == 'outgoing']
    
    print(f"Number of incoming speeds: {len(incoming_speeds)}")
    print(f"Number of outgoing speeds: {len(outgoing_speeds)}")
    
    all_speeds = incoming_speeds + outgoing_speeds
    bins = np.linspace(min(all_speeds), max(all_speeds), 20)
    
    plt.hist([incoming_speeds, outgoing_speeds], 
             bins=bins,
             label=['Incoming', 'Outgoing'],
             alpha=0.7,
             color=['#2ecc71', '#e74c3c'])
    
    plt.title('Vehicle Speed Distribution', fontsize=14, pad=20)
    plt.xlabel('Speed (km/h)', fontsize=12)
    plt.ylabel('Number of Vehicles', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.axvline(np.mean(incoming_speeds), color='#2ecc71', linestyle='--', 
                label=f'Mean Incoming: {np.mean(incoming_speeds):.1f} km/h')
    plt.axvline(np.mean(outgoing_speeds), color='#e74c3c', linestyle='--',
                label=f'Mean Outgoing: {np.mean(outgoing_speeds):.1f} km/h')
    
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "speed_distribution.png")
    print(f"Saving plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    incoming_times = [(event['frame'], event['speed_kph']) 
                     for event in data['speed_events'] if event['direction'] == 'incoming']
    outgoing_times = [(event['frame'], event['speed_kph']) 
                     for event in data['speed_events'] if event['direction'] == 'outgoing']
    
    if incoming_times:
        frames_in, speeds_in = zip(*incoming_times)
        plt.scatter(frames_in, speeds_in, label='Incoming', color='#2ecc71', alpha=0.6)
    if outgoing_times:
        frames_out, speeds_out = zip(*outgoing_times)
        plt.scatter(frames_out, speeds_out, label='Outgoing', color='#e74c3c', alpha=0.6)
    
    plt.title('Vehicle Speeds Over Time', fontsize=14, pad=20)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Speed (km/h)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "speed_over_time.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("All visualizations have been saved to the analysis directory.")

def format_traffic_analysis(data):

    total_incoming = data['total_incoming_vehicles']
    total_outgoing = data['total_outgoing_vehicles']
    speed_events = data['speed_events']
    
    incoming_speeds = [event['speed_kph'] for event in speed_events if event['direction'] == 'incoming']
    outgoing_speeds = [event['speed_kph'] for event in speed_events if event['direction'] == 'outgoing']
    
    avg_incoming_speed = sum(incoming_speeds) / len(incoming_speeds) if incoming_speeds else 0
    avg_outgoing_speed = sum(outgoing_speeds) / len(outgoing_speeds) if outgoing_speeds else 0
    
    incoming_speed_std = np.std(incoming_speeds) if incoming_speeds else 0
    outgoing_speed_std = np.std(outgoing_speeds) if outgoing_speeds else 0
    
    left_vehicles = sum(1 for event in speed_events if event['position'] == 'left')
    right_vehicles = sum(1 for event in speed_events if event['position'] == 'right')
    
    prompt = f"""Please analyze this traffic data and provide insights:

Traffic Analysis Summary:
-------------------------

Total Vehicles Detected:
  - Incoming: {total_incoming}
  - Outgoing: {total_outgoing}

Lane Information:
  - The left lane corresponds to outgoing traffic.
  - The right lane corresponds to incoming traffic.

Speed Analysis (Line Crossing Events):
-----------------------------------

Speed Analysis:
- Average incoming speed: {avg_incoming_speed:.1f} km/h (std: {incoming_speed_std:.1f})
- Average outgoing speed: {avg_outgoing_speed:.1f} km/h (std: {outgoing_speed_std:.1f})
- Number of speed measurements: {len(speed_events)}
- Speed difference (outgoing - incoming): {avg_outgoing_speed - avg_incoming_speed:.1f} km/h

Lane Usage:
- Vehicles on left side: {left_vehicles} ({left_vehicles/(left_vehicles+right_vehicles)*100:.1f}%)
- Vehicles on right side: {right_vehicles} ({right_vehicles/(left_vehicles+right_vehicles)*100:.1f}%)

Please provide:
1. A comprehensive summary of traffic patterns
2. Detailed analysis of speed distributions and variations
3. Insights about lane usage and preferences
4. Identification of any notable anomalies or patterns
5. Specific recommendations for traffic management

Format the response in a clear, structured way with sections and bullet points where appropriate.
Include specific numerical insights and their implications."""

    return prompt

def analyze_with_gemini(prompt):
    try:
        genai.configure(api_key='AIzaSyAQA-pdDgY5IoRZbnkOHHTHAw_7koNOZxE')
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"Error analyzing with Gemini: {str(e)}"

def save_analysis(analysis_text):

    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)
    output_file = os.path.join(ANALYSIS_DIR, f"traffic_analysis.txt")
    
    with open(output_file, 'w') as f:
        f.write(analysis_text)
    
    return output_file

def main():
    try:
        if not os.path.exists(ANALYSIS_DIR):
            print(f"Creating analysis directory: {ANALYSIS_DIR}")
            os.makedirs(ANALYSIS_DIR)
            
        print("Loading traffic data...")
        data = load_traffic_data()
        print("Data loaded successfully!")
        

        print("\nCreating speed distribution histogram...")
        create_visualizations(data)
        
        prompt = format_traffic_analysis(data)
        print("\nAnalyzing traffic data with Gemini...")
        analysis = analyze_with_gemini(prompt)
        
        output_file = save_analysis(analysis)
        print(f"\nAnalysis complete! Results saved to: {output_file}")
        print(f"Speed distribution histogram saved as: {os.path.join(ANALYSIS_DIR, 'speed_distribution.png')}")
        
        print("\nTraffic Analysis Results:")
        print("=" * 50)
        print(analysis)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 