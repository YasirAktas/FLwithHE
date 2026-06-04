import streamlit as st
import pandas as pd
import os
import time
import subprocess
import sys
import re
import math

st.set_page_config(page_title="FL+HE Command Center", layout="wide")

# --- CSS Injection for Figma Styling ---
st.markdown("""
    <style>
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #00D2FF 0%, #007BFF 100%);
        color: white; font-weight: 700; border: none; border-radius: 8px; transition: all 0.3s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5); transform: translateY(-1px);
    }
    button[kind="secondary"] {
        border-color: #EF4444 !important; color: #EF4444 !important; background-color: transparent !important; border-radius: 8px;
    }
    button[kind="secondary"]:hover {
        background-color: rgba(239, 68, 68, 0.1) !important; box-shadow: 0 0 10px rgba(239, 68, 68, 0.3);
    }
    div[data-testid="metric-container"] {
        background-color: #1A202C; border: 1px solid #2D3748; padding: 15px 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00D2FF, #9D4EDD); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎛️ FL+HE Command Center")

# --- 1. Session State Initialization ---
if 'process' not in st.session_state:
    st.session_state.process = None

ui_output_csv = "results/ui_live_run.csv"
ui_output_log = "results/ui_live_run.log"
os.makedirs("results", exist_ok=True)

# --- 2. Sidebar: Experiment Controls ---
st.sidebar.header("🚀 Launch Experiment")

with st.sidebar.form("experiment_form"):
    st.subheader("Training Parameters")
    dataset = st.selectbox("Dataset", ["mnist", "cifar10", "ptbxl"])
    num_clients = st.number_input("Number of Clients", min_value=1, max_value=20, value=5)
    rounds = st.number_input("Global Rounds", min_value=1, max_value=1000, value=10)
    
    st.subheader("Encryption Parameters")
    use_encryption = st.checkbox("Enable Encryption", value=True)
    scheme = st.selectbox("Encryption Scheme", ["paillier", "ckks"])
    payload = st.selectbox("Payload Mode", ["analytics", "integer_stats", "full_model"])
    
    start_button = st.form_submit_button("▶️ Start Experiment")

st.sidebar.divider()
st.sidebar.header("🛑 Process Control")
stop_button = st.sidebar.button("⏹️ Stop Current Experiment")
live_mode = st.sidebar.checkbox("Enable Live Chart Refresh", value=True)

# --- 3. Process Execution Logic ---
if start_button:
    if st.session_state.process is not None and st.session_state.process.poll() is None:
        st.sidebar.error("An experiment is already running! Stop it first.")
    else:
        if os.path.exists(ui_output_csv): os.remove(ui_output_csv)
        if os.path.exists(ui_output_log): os.remove(ui_output_log)
            
        cmd = [
            sys.executable, "-u", "-m", "src.fl.fedavg_runner", 
            "--dataset", dataset, 
            "--num_clients", str(num_clients), 
            "--rounds", str(rounds), 
            "--payload_mode", payload, 
            "--save_metrics_csv", ui_output_csv
        ]
        if use_encryption: cmd.extend(["--use_encryption", "--encryption_scheme", scheme])
            
        # FIXED: Safely open, pass to subprocess, and immediately close to prevent Windows locking
        log_file = open(ui_output_log, "w")
        st.session_state.process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        log_file.close() 
        
        st.sidebar.success("Experiment started!")

if stop_button:
    if st.session_state.process is not None and st.session_state.process.poll() is None:
        st.session_state.process.terminate()
        st.session_state.process = None
        st.sidebar.warning("Experiment stopped forcefully.")

# --- Helper Function: Draw the Network Circle ---
def generate_network_html(n_clients, client_data, current_action):
    html = f"""
    <div style="position: relative; width: 100%; height: 420px; background-color: #0B0E14; border-radius: 15px; border: 1px solid #2D3748; overflow: hidden; display: flex; align-items: center; justify-content: center; font-family: sans-serif;">
        <div style="position: absolute; text-align: center; z-index: 10;">
            <div style="font-size: 40px; text-shadow: 0 0 20px #9D4EDD;">🗄️</div>
            <div style="color: #E2E8F0; font-size: 12px; font-weight: bold; margin-top: 5px;">Global Server</div>
            <div style="color: #00D2FF; font-size: 11px; margin-top: 2px;">{current_action}</div>
        </div>
    """
    
    radius = 130 
    for i in range(n_clients):
        angle = math.radians((i * (360 / n_clients)) - 90)
        x_offset = math.cos(angle) * radius
        y_offset = math.sin(angle) * radius
        
        is_done = i in client_data
        time_text = f"{client_data[i]:.2f}s" if is_done else "..."
        
        color = "#00D2FF" if is_done else "#4A5568"
        glow = "box-shadow: 0 0 15px #00D2FF;" if is_done else ""
        icon_opacity = "1.0" if is_done else "0.4"
        
        html += f"""
        <svg style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">
            <line x1="50%" y1="50%" x2="calc(50% + {x_offset}px)" y2="calc(50% + {y_offset}px)" stroke="{color}" stroke-width="2" opacity="{icon_opacity}" stroke-dasharray="{"0" if is_done else "4"}" />
        </svg>
        <div style="position: absolute; left: calc(50% + {x_offset}px); top: calc(50% + {y_offset}px); transform: translate(-50%, -50%); text-align: center; width: 60px;">
            <div style="font-size: 24px; opacity: {icon_opacity}; border-radius: 50%; {glow} transition: all 0.3s ease; display: inline-block;">💻</div>
            <div style="color: {color}; font-size: 10px; font-weight: bold; margin-top: 4px;">Client {i}</div>
            <div style="color: #A0AEC0; font-size: 9px; font-family: monospace;">{time_text}</div>
        </div>
        """
        
    html += "</div>"
    return html

# --- 4. Live Mid-Round Progress Tracker ---
if st.session_state.process is not None and st.session_state.process.poll() is None:
    st.subheader("⏳ Live Round Topology")
    
    current_action = "Initializing..."
    current_round = 1
    completed_clients = {} 
    
    if os.path.exists(ui_output_log):
        with open(ui_output_log, "r") as f:
            logs = f.readlines()
            
        for line in logs:
            round_match = re.search(r"Round (\d+)", line)
            if round_match:
                r = int(round_match.group(1))
                if r > current_round:
                    current_round = r
                    completed_clients.clear() 
            
            client_match = re.search(r"client (\d+) training complete .* train_time=([0-9.]+)s", line)
            if client_match:
                c_id = int(client_match.group(1))
                t_time = float(client_match.group(2))
                completed_clients[c_id] = t_time
                current_action = "Training local models..."
                
            elif "encrypting" in line.lower():
                current_action = "Encrypting parameters..."
            elif "aggregation" in line.lower():
                current_action = "HE Aggregation Phase"
            elif "Acc=" in line and "Loss=" in line:
                current_action = "Global Evaluation..."

    # FIXED: Reverted to component renderer to cleanly process complex SVG/HTML without breaking Markdown
    st.components.v1.html(generate_network_html(num_clients, completed_clients, current_action), height=450)
    
elif st.session_state.process is not None:
    # FIXED: Added Crash Detector
    if st.session_state.process.poll() != 0:
        st.error("⚠️ The background experiment crashed! Here is the terminal error:")
        if os.path.exists(ui_output_log):
            with open(ui_output_log, "r") as f:
                st.code(f.read(), language="text")
    else:
        st.success("✅ **Status:** Experiment Finished!")
else:
    st.info("💤 **Status:** Idle. Ready to launch an experiment.")

st.divider()

# --- 5. Completed Rounds Data Viewer (From CSV) ---
@st.cache_data(ttl=1) 
def load_ui_data(filepath):
    if not os.path.exists(filepath): return pd.DataFrame()
    try: return pd.read_csv(filepath)
    except pd.errors.EmptyDataError: return pd.DataFrame()

df = load_ui_data(ui_output_csv)

if not df.empty:
    chart_df = df.drop_duplicates(subset=['round', 'scheme'], keep='last')
    latest = chart_df.iloc[-1] 

    st.subheader(f"Completed Metrics: Round {int(latest['round'])} / {rounds}")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric("Accuracy", f"{latest['accuracy'] * 100:.2f}%")
    kpi2.metric("Loss", f"{latest['loss']:.4f}")
    
    prev_time = chart_df.iloc[-2]['total_round_time'] if len(chart_df) > 1 else latest['total_round_time']
    time_delta = latest['total_round_time'] - prev_time
    kpi3.metric("Round Duration", f"{latest['total_round_time']:.2f}s", delta=f"{time_delta:.2f}s", delta_color="inverse")
    kpi4.metric("Payload Size", f"{latest['payload_nbytes'] / 1024:.2f} KB")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.write("**Model Convergence (Accuracy & Loss)**")
        acc_loss_df = chart_df[['round', 'accuracy', 'loss']].set_index('round')
        st.line_chart(acc_loss_df)

    with col_right:
        st.write("**Time Bottleneck Breakdown (Seconds)**")
        time_breakdown_df = chart_df[['round', 'training_time', 'encrypt_time', 'aggregate_time', 'decrypt_time']].set_index('round')
        st.bar_chart(time_breakdown_df)

# --- 6. Auto-Refresh Logic ---
if live_mode and (st.session_state.process is not None and st.session_state.process.poll() is None):
    time.sleep(1)
    st.rerun()