# Cloud Optimizer & Predictive Auto-Scaler
### Autonomous Time-Series Machine Learning Engine for Cloud Elasticity & Cost Optimization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Chart.js-success.svg)](https://flask.palletsprojects.com/)
[![Cloud Provider](https://img.shields.io/badge/Cloud-AWS%20EC2%20(ap--south--1)-orange.svg)](https://aws.amazon.com/)
[![ML Engine](https://img.shields.io/badge/ML%20Engine-Gradient%20Boosting%20(R%C2%B2%3D0.846)-purple.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Executive Overview
Modern cloud applications suffer from either **costly over-provisioning** (wasting 40–60% on idle compute) or **reactive lag** (servers crashing during sudden traffic bursts because conventional auto-scalers wait for thresholds to breach before booting nodes 2–3 minutes later).

**Cloud Optimizer** solves this dilemma through **proactive predictive scaling**. Using an ensemble **Gradient Boosting** machine learning pipeline trained on multi-variate cloud workload traces, the system forecasts CPU and resource demand (T+1) steps in advance. It autonomously provisions or decommissions worker nodes before traffic surges cause downtime, while strictly enforcing high availability and anti-flapping stability.

---

## Key Architectural Capabilities

1. **Time-Series Machine Learning Engine**:
   - Trained on 4,000 multi-variate telemetry records (modeled on Alibaba Cloud Cluster traces).
   - 19 engineered temporal features: time lags (t-1, t-2, t-3), rolling means, volatility, workload velocity, acceleration, cross-resource interactions, and 24-hour cyclical Fourier encodings.
   - Outperforms Linear Regression and Random Forest with **R² = 84.6%** and **MAE = 4.23%**.

2. **Dual-Mode AWS EC2 Auto-Scaler (`aws_scale.py`)**:
   - **Live AWS Mode**: Uses `boto3` to provision and decommission real Amazon EC2 instances in `ap-south-1` (Mumbai).
   - **High-Fidelity Simulator Mode**: Automatically activates if AWS credentials are not configured, allowing anyone (team members, evaluators) to clone and test the entire fleet lifecycle offline.

3. **180s Anti-Flapping Cooldown Buffer**:
   - Enforces a 3-minute stabilization window and a wide deadband (30% to 75%) to prevent costly oscillation and thrashing.

4. **Real-World Workload Presets**:
   - **High Traffic Rush (Exam Results / Flash Sale)**: Simulates 10,900 concurrent students hitting the portal; CPU surges to 89.4%; AI executes proactive horizontal scale-out.
   - **Night Time / Low Traffic (3:00 AM)**: Simulates 170 students online; CPU drops to 16.2%; AI decommissions surplus nodes while preserving 1 baseline standby server.
   - **Normal Daytime Traffic**: Balanced operation at ~46% CPU.

5. **Dual Telemetry Ingestion**:
   - **Laptop Hardware**: Streams real laptop processor CPU load live via `psutil`. Includes a hardware stress testing trigger (`Simulate CPU Spike`) that runs a 6-second calculation loop across CPU cores.
   - **Alibaba Cloud Cluster Dataset**: Replays 4,000 multi-variate cloud trace records.

6. **Executive Observability & Audit Trail**:
   - Real-time Chart.js dual-line graph (Solid blue: Actual CPU, Dashed amber: AI Forecast).
   - Dynamic AWS Server Rack showing active instance IDs, IPs, and terminate controls.
   - One-click scaling audit log export (`CSV Download`).
   - Printable Executive Cloud Audit Certificate with cost savings calculation (`PDF Report`).

---

## Model Performance & Benchmarks

The predictive pipeline was benchmarked using strict chronological out-of-time splits (no data leakage):

| Model Architecture | R² Score (Accuracy) | MAE (Mean Error) | RMSE | Selected Status |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **0.8462 (84.6%)** | **4.2359%** | **5.8246%** | **Champion Model** |
| Random Forest Regressor | 0.8459 (84.5%) | 4.2271% | 5.8313% | Benchmarked |
| Linear Regression | 0.7800 (78.0%) | 5.0926% | 6.9674% | Baseline |

---

## Quick Start & Setup

### Prerequisites
- Python 3.10 or higher
- Optional: AWS CLI configured (`aws configure`) for real AWS EC2 orchestration.

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/cloud-optimizer.git
cd cloud-optimizer
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python app_flask.py
```
Open your browser to:
**`http://localhost:5000`**

---

## Sharing with Team Members & Remote Access

### Option A: Local Network (Same Wi-Fi / Hotspot)
Flask binds to `0.0.0.0:5000`. Any team member connected to the same Wi-Fi can open the dashboard directly on their mobile phone or laptop using your machine's local IP:
```
http://<YOUR_LOCAL_IP>:5000
```
*(Example: `http://192.168.0.118:5000`)*

### Option B: Free Public Link via Cloudflare / Localtunnel
To share a public HTTPS link with remote team members or examiners without port-forwarding:
```bash
npx localtunnel --port 5000
# or
cloudflared tunnel --url http://localhost:5000
```

---

## Docker Deployment

The application includes a production-ready, lightweight Dockerfile:

```bash
# Build Docker image
docker build -t cloud-optimizer .

# Run container
docker run -p 5000:5000 cloud-optimizer
```

---

## Repository Structure
```
├── app_flask.py                 # RESTful Flask backend & Telemetry API
├── aws_scale.py                 # Auto-Scaling engine with boto3 AWS & Simulator
├── ml_model.py                  # Production inference pipeline & rolling buffer
├── feature_engineering.py       # 19-feature temporal pipeline
├── generate_dataset.py          # Alibaba Cloud trace generator (4,000 records)
├── train_models.py              # ML benchmarking script & chart generator
├── Dockerfile                   # Python 3.11-slim container spec
├── requirements.txt             # Pinned production dependencies
├── main.tf                      # Terraform infrastructure configuration
├── PROJECT_REPORT_AND_VIVA_GUIDE.md # 15-page academic project documentation & viva Q&A
├── templates/
│   ├── index.html               # Executive dark-mode operations dashboard
│   └── report.html              # Printable Cloud Cost Audit Certificate
├── static/images/               # 4 publication-quality ML evaluation charts
└── models/                      # Serialized models & benchmark metrics JSON
```

---

## License
This project is open for educational and research use.
