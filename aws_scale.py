"""
Intelligent Predictive Auto-Scaling & Cost Optimization Engine
Features:
- Predictive Auto-Scaling Decision Logic (Proactive vs Reactive)
- Anti-Flapping Cooldown Management (Prevents rapid launch/kill cycles)
- Cloud Financial Cost Optimization Tracker (Calculates dollar savings vs static peak provisioning)
- Dual-Mode Operation: Live AWS EC2 (boto3) with Automatic High-Fidelity Simulation Fallback
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
import uuid
from datetime import datetime

# Fleet configuration
MIN_INSTANCES = 1
MAX_INSTANCES = 4
SCALE_UP_THRESHOLD = 75.0      # Proactive trigger: Scale before saturation
SCALE_DOWN_THRESHOLD = 30.0    # Cost-saving trigger: Remove idle capacity
COOLDOWN_SECONDS = 180         # 3-minute stabilization buffer against flapping
INSTANCE_TYPE = "t3.micro"
HOURLY_COST_PER_INSTANCE = 0.0104  # AWS t3.micro on-demand pricing in ap-south-1 ($0.0104/hr)

class CloudFleetManager:
    def __init__(self, region="ap-south-1"):
        self.region = region
        self.last_scaling_action_time = 0
        self.last_action_type = "None"
        self.is_live_aws = False
        self.ec2_client = None
        
        # Financial analytics tracking
        self.simulation_start_time = time.time()
        self.accumulated_scaled_cost = 0.0
        self.accumulated_baseline_cost = 0.0
        self.scaling_events = []
        
        # Try initializing AWS EC2 Client
        self._init_aws()
        
        # Simulated Fleet State (fallback or local testing)
        self.simulated_instances = [
            {
                "InstanceId": "i-09f1a23c4d5e6b7a1",
                "InstanceType": INSTANCE_TYPE,
                "State": "running",
                "LaunchTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "PrivateIp": "172.31.24.110",
                "Tag": "AI-Scaler"
            }
        ]
        
    def _init_aws(self):
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            client = boto3.client('ec2', region_name=self.region)
            # Test connection with a lightweight call
            client.describe_instance_type_offerings(MaxResults=5)
            self.ec2_client = client
            self.is_live_aws = True
            print("[AWS ENGINE] Successfully connected to Live AWS EC2 in", self.region)
        except Exception as e:
            self.is_live_aws = False
            self.ec2_client = None
            print("[AWS ENGINE] Running in High-Fidelity EC2 Simulator Mode (AWS credentials not active or offline)")

    def get_cooldown_remaining(self):
        elapsed = time.time() - self.last_scaling_action_time
        remaining = int(COOLDOWN_SECONDS - elapsed)
        return max(0, remaining)
        
    def get_active_instances(self):
        """Returns list of active/running instances from AWS or Simulator."""
        if self.is_live_aws and self.ec2_client:
            try:
                response = self.ec2_client.describe_instances(
                    Filters=[
                        {'Name': 'tag:Project', 'Values': ['AI-Scaler']},
                        {'Name': 'instance-state-name', 'Values': ['pending', 'running']}
                    ]
                )
                instances = []
                for res in response.get('Reservations', []):
                    for inst in res.get('Instances', []):
                        instances.append({
                            "InstanceId": inst["InstanceId"],
                            "InstanceType": inst.get("InstanceType", INSTANCE_TYPE),
                            "State": inst["State"]["Name"],
                            "LaunchTime": inst.get("LaunchTime", "").strftime("%Y-%m-%d %H:%M:%S") if hasattr(inst.get("LaunchTime"), "strftime") else str(inst.get("LaunchTime")),
                            "PrivateIp": inst.get("PrivateIpAddress", "172.31.x.x"),
                            "Tag": "AI-Scaler"
                        })
                return instances
            except Exception as e:
                print(f"[AWS WARNING] Failed to query AWS EC2: {e}. Falling back to simulator.")
                self.is_live_aws = False
                
        # Simulated Fleet
        return [i for i in self.simulated_instances if i["State"] in ["running", "pending"]]

    def evaluate_and_scale(self, current_cpu, predicted_cpu, bypass_cooldown=False):
        """
        Executes the intelligent predictive scaling decision pipeline:
        1. Checks anti-flapping cooldown timer (unless bypass_cooldown is True).
        2. Evaluates proactive scaling thresholds based on predicted future load and current surge.
        3. Updates cloud financial savings metrics.
        """
        active_instances = self.get_active_instances()
        instance_count = len(active_instances)
        cooldown_remaining = self.get_cooldown_remaining()
        
        # 1. Evaluate Decision: Proactive scale up triggers if EITHER forecast or current load breaches threshold
        effective_scale_up_load = max(current_cpu, predicted_cpu)
        
        if effective_scale_up_load >= SCALE_UP_THRESHOLD:
            decision = "PROACTIVE SCALE-OUT"
            status_class = "scale-up"
            if predicted_cpu >= SCALE_UP_THRESHOLD:
                rationale = f"Forecasted CPU ({predicted_cpu:.1f}%) breaches SLA threshold ({SCALE_UP_THRESHOLD}%). Provisioning additional EC2 worker node proactively."
            else:
                rationale = f"Observed workload surge ({current_cpu:.1f}%) exceeds safety threshold ({SCALE_UP_THRESHOLD}%). Triggering immediate horizontal scale-out."
            
            if cooldown_remaining > 0 and not bypass_cooldown:
                action_msg = f"Cooldown lock active ({cooldown_remaining}s remaining). Provisioning deferred to maintain fleet stability."
            elif instance_count >= MAX_INSTANCES:
                action_msg = f"Cluster ceiling reached ({MAX_INSTANCES} nodes max). Operating at maximum provisioned capacity."
            else:
                action_msg = self._trigger_scale_up()
                self.last_scaling_action_time = time.time()
                self.last_action_type = "Scale Out"
                
        elif current_cpu <= SCALE_DOWN_THRESHOLD and predicted_cpu <= SCALE_DOWN_THRESHOLD:
            decision = "COST-OPTIMIZED SCALE-IN"
            status_class = "scale-down"
            rationale = f"Forecasted CPU ({predicted_cpu:.1f}%) and current load ({current_cpu:.1f}%) are below idle threshold ({SCALE_DOWN_THRESHOLD}%). Decommissioning surplus capacity."
            
            if cooldown_remaining > 0 and not bypass_cooldown:
                action_msg = f"Cooldown lock active ({cooldown_remaining}s remaining). Scale-in deferred."
            elif instance_count <= MIN_INSTANCES:
                action_msg = f"Baseline cluster limit maintained ({MIN_INSTANCES} node). Preserving primary node to ensure 100% availability SLA."
            else:
                action_msg = self._trigger_scale_down(active_instances)
                self.last_scaling_action_time = time.time()
                self.last_action_type = "Scale In"
                
        else:
            decision = "STEADY STATE (BALANCED)"
            status_class = "normal"
            rationale = f"Telemetry ({current_cpu:.1f}%) and forecast ({predicted_cpu:.1f}%) reside within nominal operational headroom [{SCALE_DOWN_THRESHOLD}% - {SCALE_UP_THRESHOLD}%]."
            action_msg = "No scaling action required. Fleet capacity is optimally provisioned."
            
        # Log to DevOps Audit Trail
        current_active = len(self.get_active_instances())
        self._log_event(decision, current_cpu, predicted_cpu, action_msg, current_active, rationale)
            
        # 2. Update Financial Metrics
        fin_stats = self._calculate_financials(len(self.get_active_instances()))
        
        return {
            "decision": decision,
            "action": action_msg,
            "status": status_class,
            "rationale": rationale,
            "active_instances": len(self.get_active_instances()),
            "cooldown_remaining": self.get_cooldown_remaining(),
            "cost_stats": fin_stats,
            "mode": "Live AWS EC2 (boto3)" if self.is_live_aws else "High-Fidelity Cloud Fleet Simulator"
        }
        
    def _trigger_scale_up(self):
        if self.is_live_aws and self.ec2_client:
            try:
                res = self.ec2_client.run_instances(
                    ImageId='ami-0f58b397bc5c1f2e8', # Ubuntu 22.04 LTS ap-south-1
                    InstanceType=INSTANCE_TYPE,
                    MinCount=1,
                    MaxCount=1,
                    TagSpecifications=[{
                        'ResourceType': 'instance',
                        'Tags': [{'Key': 'Project', 'Value': 'AI-Scaler'}]
                    }]
                )
                inst_id = res['Instances'][0]['InstanceId']
                return f"🚀 Real AWS EC2 Instance Created: {inst_id} (Region: {self.region}). Visible in your AWS Console now!"
            except Exception as e:
                return f"AWS EC2 Launch Error: {str(e)}"
                
        # Simulator provision
        new_id = f"i-{uuid.uuid4().hex[:17]}"
        oct4 = 100 + len(self.simulated_instances) * 15
        new_inst = {
            "InstanceId": new_id,
            "InstanceType": INSTANCE_TYPE,
            "State": "running",
            "LaunchTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "PrivateIp": f"172.31.24.{oct4}",
            "Tag": "AI-Scaler"
        }
        self.simulated_instances.append(new_inst)
        return f"Simulated EC2 Instance Provisioned ({new_id})"
        
    def _trigger_scale_down(self, active_instances):
        if not active_instances:
            return "No active instances available to terminate."
            
        target_instance_id = active_instances[-1]["InstanceId"]
        
        if self.is_live_aws and self.ec2_client:
            try:
                self.ec2_client.terminate_instances(InstanceIds=[target_instance_id])
                return f"📉 Real AWS EC2 Instance Terminated: {target_instance_id} (Region: {self.region}). Check your AWS Console now!"
            except Exception as e:
                return f"AWS EC2 Termination Error: {str(e)}"
                
        # Simulator decommission
        for inst in self.simulated_instances:
            if inst["InstanceId"] == target_instance_id:
                inst["State"] = "terminated"
                break
        return f"Simulated EC2 Instance Decommissioned ({target_instance_id})"

    def terminate_specific_instance(self, target_instance_id):
        if self.is_live_aws and self.ec2_client:
            try:
                self.ec2_client.terminate_instances(InstanceIds=[target_instance_id])
                return True, f"Successfully terminated AWS EC2 Instance {target_instance_id} in your AWS account."
            except Exception as e:
                return False, f"AWS Termination Error: {str(e)}"
        
        for inst in self.simulated_instances:
            if inst["InstanceId"] == target_instance_id:
                inst["State"] = "terminated"
                return True, f"Simulated instance {target_instance_id} terminated."
        return False, f"Instance {target_instance_id} not found."
        
    def _log_event(self, event_type, current_cpu, forecast_cpu, action_taken, fleet_size, reason):
        entry = {
            "id": len(self.scaling_events) + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "event_type": event_type,
            "current_cpu": f"{float(current_cpu):.1f}%",
            "forecast_cpu": f"{float(forecast_cpu):.1f}%",
            "action": action_taken,
            "fleet_size": f"{fleet_size} / {MAX_INSTANCES}",
            "reason": reason
        }
        self.scaling_events.insert(0, entry)
        if len(self.scaling_events) > 50:
            self.scaling_events.pop()

    def _calculate_financials(self, current_fleet_count):
        """Calculates cost savings achieved through dynamic scaling versus static peak provisioning."""
        # Static baseline: paying for MAX_INSTANCES 24/7 to guarantee peak capacity
        baseline_hourly = MAX_INSTANCES * HOURLY_COST_PER_INSTANCE
        baseline_monthly = baseline_hourly * 730 # 730 hours in average month ($30.37/mo)
        
        # Dynamic managed cost
        current_hourly = current_fleet_count * HOURLY_COST_PER_INSTANCE
        current_monthly = current_hourly * 730
        
        hourly_savings = max(0.0, baseline_hourly - current_hourly)
        monthly_savings = max(0.0, baseline_monthly - current_monthly)
        savings_percent = (hourly_savings / baseline_hourly) * 100.0 if baseline_hourly > 0 else 0.0
        
        return {
            "current_hourly_cost": f"${current_hourly:.4f}/hr",
            "baseline_hourly_cost": f"${baseline_hourly:.4f}/hr",
            "monthly_savings_est": f"${monthly_savings:.2f}/mo",
            "savings_percent": f"{savings_percent:.1f}%",
            "active_nodes": current_fleet_count,
            "max_nodes": MAX_INSTANCES,
            "instance_flavor": INSTANCE_TYPE
        }

# Singleton fleet manager
fleet_manager = CloudFleetManager()

def scale_up():
    """Legacy helper for backward compatibility."""
    return fleet_manager._trigger_scale_up()

def scale_down():
    """Legacy helper for backward compatibility."""
    active = fleet_manager.get_active_instances()
    return fleet_manager._trigger_scale_down(active)

if __name__ == "__main__":
    print("\n--- Testing Cloud Fleet Manager ---")
    active = fleet_manager.get_active_instances()
    print("Active Instances:", active)
    print("\n--- Simulating Decision (Predicted CPU: 88%) ---")
    eval_res = fleet_manager.evaluate_and_scale(current_cpu=65.0, predicted_cpu=88.0)
    print("Decision:", eval_res["decision"])
    print("Action:", eval_res["action"])
    print("Financials:", eval_res["cost_stats"])
