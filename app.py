# app.py
from flask import Flask, request, jsonify, render_template
import subprocess
import pandas as pd
import os
import sys
import re

app = Flask(__name__)

# State management (similar to st.session_state)
process = None
ui_output_csv = "results/ui_live_run.csv"
ui_output_log = "results/ui_live_run.log"

@app.route('/')
def index():
    # Serve the main HTML page
    return render_template('index.html')

# Inside app.py - Update the /api/start route
@app.route('/api/start', methods=['POST'])
def start_experiment():
    global process
    if process is not None and process.poll() is None:
        return jsonify({"status": "error", "message": "Experiment already running"}), 400

    data = request.json
    os.makedirs("results", exist_ok=True)
    if os.path.exists(ui_output_csv): os.remove(ui_output_csv)
    if os.path.exists(ui_output_log): os.remove(ui_output_log)

    cmd = [
        sys.executable, "-u", "-m", "src.fl.fedavg_runner",
        "--dataset", data.get('dataset', 'mnist').split(" ")[0].lower(), # Parse "MNIST (Handwritten)" -> "mnist"
        "--num_clients", str(data.get('num_clients', 5)),
        "--rounds", str(data.get('rounds', 10)),
        "--payload_mode", "full_model",
        "--save_metrics_csv", ui_output_csv
    ]
    
    # Map the React "Privacy Method" to actual flags
    method = data.get('method', 'Hybrid')
    
    # DP Flags
    if method in ['DP Only', 'Hybrid']:
        cmd.extend([
            "--use_dp",
            "--dp_mechanism", data.get('dp_mechanism', 'Gaussian Noise').split(" ")[0].lower(),
            "--dp_epsilon", str(data.get('epsilon', 2.0))
        ])
        
    # HE Flags
    if method in ['HE Only', 'Hybrid']:
        scheme = data.get('scheme', 'CKKS').split(" ")[0].lower() # Parse "CKKS (Approximate)" -> "ckks"
        cmd.extend(["--use_encryption", "--encryption_scheme", scheme])

    log_file = open(ui_output_log, "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    log_file.close()

    return jsonify({"status": "success", "message": "Experiment started"})

@app.route('/api/stop', methods=['POST'])
def stop_experiment():
    global process
    if process is not None and process.poll() is None:
        process.terminate()
        process = None
        return jsonify({"status": "success", "message": "Experiment stopped"})
    return jsonify({"status": "idle"})

@app.route('/api/status', methods=['GET'])
def get_status():
    global process
    
    is_running = process is not None and process.poll() is None
    crashed = process is not None and process.poll() != 0 and process.poll() is not None
    
    current_action = "Idle"
    completed_clients = {}
    
    if is_running and os.path.exists(ui_output_log):
        with open(ui_output_log, "r") as f:
            lines = f.readlines()
            for line in lines:
                client_match = re.search(r"client (\d+) training complete .* train_time=([0-9.]+)s", line)
                if client_match:
                    completed_clients[int(client_match.group(1))] = float(client_match.group(2))
                    current_action = "Training local models..."
                elif "encrypting" in line.lower():
                    current_action = "Encrypting parameters..."
                elif "aggregation" in line.lower():
                    current_action = "HE Aggregation Phase"
    
    return jsonify({
        "is_running": is_running,
        "crashed": crashed,
        "current_action": current_action,
        "completed_clients": completed_clients
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if not os.path.exists(ui_output_csv):
        return jsonify({})
    
    try:
        df = pd.read_csv(ui_output_csv)
        if df.empty: return jsonify({})
        
        # Convert DataFrame to JSON for the frontend to chart
        chart_df = df.drop_duplicates(subset=['round', 'scheme'], keep='last')
        latest = chart_df.iloc[-1].to_dict()
        all_data = chart_df.to_dict(orient='records')
        
        return jsonify({
            "latest": latest,
            "history": all_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)