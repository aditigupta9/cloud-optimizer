"""
Cloud Resource Optimization System - Flask Application
RESTful Telemetry Server & Executive Operations Dashboard
Integrates:
- Real-time time-series predictive inference (Gradient Boosting / Random Forest)
- Predictive Auto-Scaling Engine with anti-flapping protection (aws_scale.py)
- University Real-World Workload Presets (Result Day Rush vs Idle Semester Day)
- Live Machine Hardware Telemetry (psutil) & Alibaba Cloud Trace Playback
- DevOps Scaling Audit Log & CSV / PDF Report Generator
"""

import os
import io
import csv
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from ml_model import (
    predict_cpu, 
    predict_future_cpu, 
    get_telemetry_record, 
    get_live_hardware_reading,
    get_model_benchmarks, 
    _telemetry_buffer
)
from aws_scale import (
    fleet_manager, 
    HOURLY_COST_PER_INSTANCE, 
    MAX_INSTANCES, 
    MIN_INSTANCES, 
    SCALE_UP_THRESHOLD, 
    SCALE_DOWN_THRESHOLD
)

app = Flask(__name__)

# State tracking
current_step = 12
telemetry_source = "live"  # "live" or "dataset"
history_timeline = []
history_actual_cpu = []
history_predicted_cpu = []

def calculate_active_students(cpu_percent, net_mbps=20.0):
    """Calculates realistic active concurrent student / user count from resource load."""
    if cpu_percent < 25.0:
        return int(cpu_percent * 8) + 45   # ~150-250 students (off-peak/idle)
    elif cpu_percent < 65.0:
        return int(cpu_percent * 45) + 320 # ~1,500 - 3,200 students (regular classes)
    else:
        return int(cpu_percent * 85) + int(net_mbps * 8) + 1200 # ~7,500 - 11,000 students (result release rush!)

def init_history():
    global current_step, history_timeline, history_actual_cpu, history_predicted_cpu
    history_timeline = []
    history_actual_cpu = []
    history_predicted_cpu = []
    
    if telemetry_source == "live":
        for i in range(6):
            rec = get_live_hardware_reading()
            _telemetry_buffer.add_reading(rec)
            cpu_val = float(rec["cpu_util_percent"])
            history_timeline.append(datetime.now().strftime("%H:%M:%S"))
            history_actual_cpu.append(cpu_val)
            history_predicted_cpu.append(cpu_val)
    else:
        for s in range(1, 9):
            rec = get_telemetry_record(s)
            if rec:
                cpu_val = float(rec["cpu_util_percent"])
                history_timeline.append(f"T+{s}")
                history_actual_cpu.append(cpu_val)
                history_predicted_cpu.append(cpu_val)
        current_step = 8

init_history()

@app.route('/')
def home():
    active_instances = fleet_manager.get_active_instances()
    fin_stats = fleet_manager._calculate_financials(len(active_instances))
    model_benchmarks = get_model_benchmarks()
    
    latest_actual = history_actual_cpu[-1] if history_actual_cpu else 25.0
    latest_predicted = history_predicted_cpu[-1] if history_predicted_cpu else 25.0
    active_students = calculate_active_students(latest_actual)
    
    return render_template(
        "index.html",
        timeline=history_timeline,
        actual_cpu=history_actual_cpu,
        predicted_cpu=history_predicted_cpu,
        current_step=current_step,
        latest_actual=latest_actual,
        latest_predicted=latest_predicted,
        active_students=f"{active_students:,}",
        instances=active_instances,
        financials=fin_stats,
        model_benchmarks=model_benchmarks,
        logs=fleet_manager.scaling_events,
        source=telemetry_source,
        mode="Live AWS EC2 (boto3)" if fleet_manager.is_live_aws else "Cloud Fleet Simulator",
        region=fleet_manager.region,
        scale_up_thresh=SCALE_UP_THRESHOLD,
        scale_down_thresh=SCALE_DOWN_THRESHOLD
    )

# --- UNIVERSITY REAL-WORLD SCENARIOS ---

@app.route('/api/trigger_scenario', methods=['POST'])
def api_trigger_scenario():
    """Simulates real-world university workloads: Result Day Rush vs Idle Semester Day vs Regular Class Day."""
    data = request.get_json() or {}
    scenario = data.get("scenario", "result_day")
    now_time = datetime.now().strftime("%H:%M:%S")
    
    if scenario == "result_day":
        title = "High Traffic Rush (Exam Results / Flash Sale)"
        for temp_cpu in [68.0, 79.0, 85.0]:
            _telemetry_buffer.add_reading({
                "time_step": int(datetime.now().timestamp()) - 30,
                "hour_of_day": 10.5,
                "cpu_util_percent": temp_cpu,
                "mem_util_percent": 75.0,
                "net_in_mbps": 180.0,
                "net_out_mbps": 320.0,
                "disk_io_percent": 60.0,
                "source": "scenario_result_day"
            })
        reading = {
            "time_step": int(datetime.now().timestamp()),
            "hour_of_day": 10.5, # 10:30 AM peak announcement window
            "cpu_util_percent": 89.4,
            "mem_util_percent": 84.5,
            "net_in_mbps": 265.0,
            "net_out_mbps": 510.0,
            "disk_io_percent": 82.0,
            "source": "scenario_result_day"
        }
        custom_reason = "Heavy traffic rush: ~10,900 students online. CPU at 89%. AI launched a new AWS server automatically to prevent website slowdown."
        
    elif scenario == "idle_day":
        title = "Night Time / Low Traffic (3:00 AM)"
        for temp_cpu in [35.0, 26.0, 20.0]:
            _telemetry_buffer.add_reading({
                "time_step": int(datetime.now().timestamp()) - 30,
                "hour_of_day": 3.0,
                "cpu_util_percent": temp_cpu,
                "mem_util_percent": 38.0,
                "net_in_mbps": 14.0,
                "net_out_mbps": 20.0,
                "disk_io_percent": 10.0,
                "source": "scenario_idle_day"
            })
        reading = {
            "time_step": int(datetime.now().timestamp()),
            "hour_of_day": 3.0, # 3:00 AM off-peak window
            "cpu_util_percent": 16.2,
            "mem_util_percent": 34.0,
            "net_in_mbps": 8.5,
            "net_out_mbps": 12.0,
            "disk_io_percent": 6.0,
            "source": "scenario_idle_day"
        }
        custom_reason = "Low night traffic: ~170 students online. CPU at 16%. AI safely turned off extra idle servers to save cloud costs, leaving 1 active."
        
    else: # regular_day
        title = "Normal Daytime Traffic"
        for temp_cpu in [42.0, 48.0, 44.0]:
            _telemetry_buffer.add_reading({
                "time_step": int(datetime.now().timestamp()) - 30,
                "hour_of_day": 14.0,
                "cpu_util_percent": temp_cpu,
                "mem_util_percent": 50.0,
                "net_in_mbps": 50.0,
                "net_out_mbps": 80.0,
                "disk_io_percent": 22.0,
                "source": "scenario_regular_day"
            })
        reading = {
            "time_step": int(datetime.now().timestamp()),
            "hour_of_day": 14.0, # 2:00 PM standard operating window
            "cpu_util_percent": 46.5,
            "mem_util_percent": 51.0,
            "net_in_mbps": 52.0,
            "net_out_mbps": 85.0,
            "disk_io_percent": 24.0,
            "source": "scenario_regular_day"
        }
        custom_reason = "Balanced daytime traffic: ~2,400 students browsing classes. CPU is stable at 46%. Normal capacity is sufficient."

    _telemetry_buffer.add_reading(reading)
    pred_res = predict_future_cpu()
    
    current_cpu = float(reading["cpu_util_percent"])
    predicted_cpu = pred_res["predicted_cpu"]
    
    # Bypass cooldown for manual interactive scenario testing so user immediately sees scaling action
    scale_res = fleet_manager.evaluate_and_scale(current_cpu, predicted_cpu, bypass_cooldown=True)
    
    # Log with university scenario context
    fleet_manager._log_event(
        scale_res["decision"], 
        current_cpu, 
        predicted_cpu, 
        scale_res["action"], 
        len(fleet_manager.get_active_instances()), 
        f"{title} - {custom_reason}"
    )
    
    history_timeline.append(now_time)
    history_actual_cpu.append(round(current_cpu, 2))
    history_predicted_cpu.append(round(predicted_cpu, 2))
    
    if len(history_timeline) > 25:
        history_timeline.pop(0)
        history_actual_cpu.pop(0)
        history_predicted_cpu.pop(0)
        
    active_students = calculate_active_students(current_cpu, reading["net_in_mbps"])
    active_instances = fleet_manager.get_active_instances()
    
    return jsonify({
        "success": True,
        "scenario": scenario,
        "scenario_title": title,
        "active_students": f"{active_students:,}",
        "step": now_time,
        "current_telemetry": {
            "cpu_percent": round(current_cpu, 2),
            "mem_percent": round(reading["mem_util_percent"], 2),
            "net_in_mbps": round(reading["net_in_mbps"], 2),
            "net_out_mbps": round(reading["net_out_mbps"], 2),
            "disk_io_percent": round(reading["disk_io_percent"], 2),
            "hour_of_day": round(reading["hour_of_day"], 2)
        },
        "prediction": pred_res,
        "scaling": scale_res,
        "fleet": active_instances,
        "logs": fleet_manager.scaling_events,
        "chart_data": {
            "labels": history_timeline,
            "actual": history_actual_cpu,
            "predicted": history_predicted_cpu
        }
    })

# --- TELEMETRY SOURCE TOGGLE ---

@app.route('/api/set_source', methods=['POST'])
def api_set_source():
    global telemetry_source
    data = request.get_json() or {}
    new_source = data.get("source", "live")
    if new_source in ["live", "dataset"]:
        telemetry_source = new_source
        init_history()
        return jsonify({"success": True, "source": telemetry_source})
    return jsonify({"success": False, "message": "Invalid source"}), 400

@app.route('/api/get_source', methods=['GET'])
def api_get_source():
    return jsonify({"source": telemetry_source})

# --- STEP SIMULATION / LIVE UPDATE ---

@app.route('/api/simulate_step', methods=['POST'])
def api_simulate_step():
    global current_step
    
    if telemetry_source == "live":
        rec = get_live_hardware_reading()
        label = datetime.now().strftime("%H:%M:%S")
        step_display = label
    else:
        current_step += 1
        rec = get_telemetry_record(current_step)
        if not rec:
            current_step = 1
            rec = get_telemetry_record(current_step)
        label = f"T+{current_step}"
        step_display = f"T+{current_step}"
        
    _telemetry_buffer.add_reading(rec)
    pred_res = predict_future_cpu()
    
    current_cpu = float(rec["cpu_util_percent"])
    predicted_cpu = pred_res["predicted_cpu"]
    
    scale_res = fleet_manager.evaluate_and_scale(current_cpu, predicted_cpu)
    
    history_timeline.append(label)
    history_actual_cpu.append(round(current_cpu, 2))
    history_predicted_cpu.append(round(predicted_cpu, 2))
    
    if len(history_timeline) > 25:
        history_timeline.pop(0)
        history_actual_cpu.pop(0)
        history_predicted_cpu.pop(0)
        
    active_students = calculate_active_students(current_cpu, rec.get("net_in_mbps", 20.0))
    active_instances = fleet_manager.get_active_instances()
    
    return jsonify({
        "success": True,
        "step": step_display,
        "source": telemetry_source,
        "active_students": f"{active_students:,}",
        "current_telemetry": {
            "cpu_percent": round(current_cpu, 2),
            "mem_percent": round(rec["mem_util_percent"], 2),
            "net_in_mbps": round(rec["net_in_mbps"], 2),
            "net_out_mbps": round(rec["net_out_mbps"], 2),
            "disk_io_percent": round(rec["disk_io_percent"], 2),
            "hour_of_day": round(rec["hour_of_day"], 2)
        },
        "prediction": pred_res,
        "scaling": scale_res,
        "fleet": active_instances,
        "logs": fleet_manager.scaling_events,
        "chart_data": {
            "labels": history_timeline,
            "actual": history_actual_cpu,
            "predicted": history_predicted_cpu
        }
    })

# --- MANUAL ACTIONS & FLEET ---

@app.route('/api/manual_scale', methods=['POST'])
def api_manual_scale():
    data = request.get_json() or {}
    action_type = data.get("action", "")
    active_instances = fleet_manager.get_active_instances()
    
    curr_cpu = history_actual_cpu[-1] if history_actual_cpu else 50.0
    pred_cpu = history_predicted_cpu[-1] if history_predicted_cpu else 50.0
    
    if action_type == "scale_up":
        if len(active_instances) >= MAX_INSTANCES:
            return jsonify({"success": False, "message": f"Maximum capacity reached ({MAX_INSTANCES} nodes)"})
        msg = fleet_manager._trigger_scale_up()
        new_active = len(fleet_manager.get_active_instances())
        fleet_manager._log_event("MANUAL SCALE UP", curr_cpu, pred_cpu, msg, new_active, "Operator manual scale up override")
        return jsonify({"success": True, "message": msg, "instances": fleet_manager.get_active_instances(), "logs": fleet_manager.scaling_events})
        
    elif action_type == "scale_down":
        if len(active_instances) <= MIN_INSTANCES:
            return jsonify({"success": False, "message": f"Minimum baseline reached ({MIN_INSTANCES} node)"})
        msg = fleet_manager._trigger_scale_down(active_instances)
        new_active = len(fleet_manager.get_active_instances())
        fleet_manager._log_event("MANUAL SCALE DOWN", curr_cpu, pred_cpu, msg, new_active, "Operator manual scale down override")
        return jsonify({"success": True, "message": msg, "instances": fleet_manager.get_active_instances(), "logs": fleet_manager.scaling_events})
        
    return jsonify({"success": False, "message": "Invalid action specified."})

@app.route('/api/terminate_instance', methods=['POST'])
def api_terminate_instance():
    data = request.get_json() or {}
    instance_id = data.get("instance_id", "")
    if not instance_id:
        return jsonify({"success": False, "message": "No instance ID provided."})
    success, msg = fleet_manager.terminate_specific_instance(instance_id)
    curr_cpu = history_actual_cpu[-1] if history_actual_cpu else 50.0
    pred_cpu = history_predicted_cpu[-1] if history_predicted_cpu else 50.0
    fleet_manager._log_event("MANUAL TERMINATION", curr_cpu, pred_cpu, msg, len(fleet_manager.get_active_instances()), f"Direct termination of {instance_id}")
    return jsonify({"success": success, "message": msg, "instances": fleet_manager.get_active_instances(), "logs": fleet_manager.scaling_events})

@app.route('/api/fleet_status', methods=['GET'])
def api_fleet_status():
    active_instances = fleet_manager.get_active_instances()
    fin_stats = fleet_manager._calculate_financials(len(active_instances))
    return jsonify({
        "mode": "Live AWS EC2 (boto3)" if fleet_manager.is_live_aws else "Cloud Fleet Simulator",
        "instances": active_instances,
        "financials": fin_stats,
        "cooldown_remaining": fleet_manager.get_cooldown_remaining(),
        "recent_events": fleet_manager.scaling_events
    })

@app.route('/api/audit_logs', methods=['GET'])
def api_audit_logs():
    return jsonify(fleet_manager.scaling_events)

@app.route('/api/model_benchmarks', methods=['GET'])
def api_model_benchmarks():
    return jsonify(get_model_benchmarks())

@app.route('/api/predict_custom', methods=['POST'])
def api_predict_custom():
    data = request.get_json() or {}
    custom_reading = {
        "time_step": int(datetime.now().timestamp()),
        "hour_of_day": float(data.get("hour_of_day", 14.0)),
        "cpu_util_percent": float(data.get("cpu_percent", 50.0)),
        "mem_util_percent": float(data.get("mem_percent", 50.0)),
        "net_in_mbps": float(data.get("net_in_mbps", 60.0)),
        "net_out_mbps": float(data.get("net_out_mbps", 100.0)),
        "disk_io_percent": float(data.get("disk_io_percent", 20.0))
    }
    _telemetry_buffer.add_reading(custom_reading)
    pred_res = predict_future_cpu()
    scale_res = fleet_manager.evaluate_and_scale(
        current_cpu=custom_reading["cpu_util_percent"],
        predicted_cpu=pred_res["predicted_cpu"]
    )
    return jsonify({
        "success": True,
        "input": custom_reading,
        "prediction": pred_res,
        "scaling": scale_res
    })

# --- REPORTS & CSV DOWNLOAD ---

@app.route('/report')
def report_page():
    active_instances = fleet_manager.get_active_instances()
    fin_stats = fleet_manager._calculate_financials(len(active_instances))
    return render_template(
        "report.html",
        financials=fin_stats,
        logs=fleet_manager.scaling_events,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route('/download/audit_csv')
def download_audit_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Event ID",
        "Timestamp",
        "Event Type",
        "Current CPU",
        "AI Forecasted CPU",
        "Fleet Nodes",
        "Action Executed",
        "Decision Rationale"
    ])
    for log in fleet_manager.scaling_events:
        writer.writerow([
            log.get("id", ""),
            log.get("timestamp", ""),
            log.get("event_type", ""),
            log.get("current_cpu", ""),
            log.get("forecast_cpu", ""),
            log.get("fleet_size", ""),
            log.get("action", ""),
            log.get("reason", "")
        ])
    csv_data = output.getvalue()
    filename = f"cloud_scaling_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

import subprocess

@app.route('/api/stress_cpu', methods=['POST'])
def api_stress_cpu():
    """Spawns an isolated background process that calculates math for 6 seconds on laptop CPU."""
    code = "import time; t=time.time()+6\nwhile time.time()<t: _=[x*x for x in range(50000)]"
    subprocess.Popen([sys.executable, "-c", code])
    return jsonify({"success": True, "message": "Real hardware CPU stress active for 6 seconds! Watch your CPU spike!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1"]
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
