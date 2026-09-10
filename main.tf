terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-south-1"
}

# --- 1. Security Group for AI Cloud Controller & Dynamic Fleet ---
resource "aws_security_group" "scaler_sg" {
  name        = "ai-scaler-security-group"
  description = "Security group for Intelligent Cloud Resource Optimization System"

  # Inbound HTTP (Port 80)
  ingress {
    description = "Allow HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Inbound Flask App (Port 5000)
  ingress {
    description = "Allow Flask Telemetry Dashboard"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Inbound SSH (Port 22)
  ingress {
    description = "Allow SSH management"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound All Traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "AI-Scaler-SG"
    Project = "AI-Scaler"
  }
}

# --- 2. Master AI Controller Instance (Hosts Flask & ML Engine) ---
resource "aws_instance" "scaler_controller" {
  ami                    = "ami-0f58b397bc5c1f2e8" # Ubuntu 22.04 LTS (ap-south-1)
  instance_type          = "t3.micro"
  vpc_security_group_ids = [aws_security_group.scaler_sg.id]

  tags = {
    Name    = "AI-Scaler-Controller"
    Project = "AI-Scaler"
    Role    = "Inference-And-Optimization-Server"
  }
}

# --- 3. EC2 Launch Template for Dynamically Scaled Worker Nodes ---
resource "aws_launch_template" "worker_template" {
  name_prefix   = "ai-scaler-worker-"
  image_id      = "ami-0f58b397bc5c1f2e8"
  instance_type = "t3.micro"

  vpc_security_group_ids = [aws_security_group.scaler_sg.id]

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "AI-Scaler-Worker"
      Project = "AI-Scaler"
      Managed = "ML-Predictive-Engine"
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# --- 4. Outputs for Easy Dashboard Access ---
output "controller_public_ip" {
  description = "Public IP address of the AI Scaler Controller"
  value       = aws_instance.scaler_controller.public_ip
}

output "dashboard_url" {
  description = "Direct web link to the AI Telemetry Dashboard"
  value       = "http://${aws_instance.scaler_controller.public_ip}:5000"
}

output "security_group_id" {
  description = "Security Group ID assigned to the fleet"
  value       = aws_security_group.scaler_sg.id
}
