provider "aws" {
  region = "ca-central-1" # Canada Central for data residency compliance
}

# ---------------------------------------------------------------------------------------------------------------------
# SECURITY & CRYPTOGRAPHY (KMS)
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_kms_key" "imaging_key" {
  description             = "KMS key for encrypting imaging data at rest"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "Sunnybrook-Imaging-KMS"
  }
}

resource "aws_kms_alias" "imaging_key_alias" {
  name          = "alias/sunnybrook-imaging-key"
  target_key_id = aws_kms_key.imaging_key.key_id
}

# ---------------------------------------------------------------------------------------------------------------------
# STORAGE (ENCRYPTED)
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_s3_bucket" "imaging_data_lake" {
  bucket = "sunnybrook-imaging-research-data-lake"

  tags = {
    Name        = "Imaging Data Lake"
    Environment = "Research"
    Security    = "KMS-Encrypted"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake_crypto" {
  bucket = aws_s3_bucket.imaging_data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.imaging_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "lake_security" {
  bucket = aws_s3_bucket.imaging_data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "bucket_config" {
  bucket = aws_s3_bucket.imaging_data_lake.id

  rule {
    id = "archive_cold_data"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

resource "aws_ecr_repository" "analysis_containers" {
  name                 = "sunnybrook-analysis-containers"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.imaging_key.arn
  }
}

resource "aws_sagemaker_notebook_instance" "research_notebook" {
  name          = "sunnybrook-research-notebook"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = "ml.t3.medium"
  kms_key_id    = aws_kms_key.imaging_key.arn
  
  # Ensure direct internet access is disabled for security if using VPC
  direct_internet_access = "Disabled"
  subnet_id              = aws_subnet.public_subnet_a.id 
  security_groups        = [aws_security_group.notebook_sg.id]
}

resource "aws_iam_role" "sagemaker_role" {
  name = "sunnybrook_sagemaker_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# ---------------------------------------------------------------------------------------------------------------------
# NETWORKING & LOAD BALANCING
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_vpc" "research_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name = "Sunnybrook Research VPC"
  }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.research_vpc.id
}

resource "aws_subnet" "public_subnet_a" {
  vpc_id                  = aws_vpc.research_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "ca-central-1a"
  map_public_ip_on_launch = true
}

resource "aws_subnet" "public_subnet_b" {
  vpc_id                  = aws_vpc.research_vpc.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "ca-central-1b"
  map_public_ip_on_launch = true
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.research_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.public_subnet_a.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "b" {
  subnet_id      = aws_subnet.public_subnet_b.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_security_group" "alb_sg" {
  name        = "sunnybrook-alb-sg"
  description = "Allow inbound HTTP traffic"
  vpc_id      = aws_vpc.research_vpc.id

  ingress {
    description = "HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # HTTPS Support
  ingress {
    description = "HTTPS from anywhere"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "notebook_sg" {
  name        = "sunnybrook-notebook-sg"
  description = "Security group for SageMaker Notebook"
  vpc_id      = aws_vpc.research_vpc.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}


resource "aws_lb" "imaging_alb" {
  name               = "sunnybrook-imaging-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = [aws_subnet.public_subnet_a.id, aws_subnet.public_subnet_b.id]

  tags = {
    Name = "Imaging Application Load Balancer"
  }
}

resource "aws_lb_target_group" "inference_nodes" {
  name     = "sunnybrook-inference-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.research_vpc.id

  health_check {
    path = "/health"
  }
}

resource "aws_lb_listener" "http_listener" {
  load_balancer_arn = aws_lb.imaging_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.inference_nodes.arn
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# AUTO SCALING
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_launch_template" "inference_lt" {
  name_prefix   = "sunnybrook-inference-"
  image_id      = "ami-0c55b159cbfafe1f0" # Example Ubuntu AMI, in real life use data source
  instance_type = "t3.medium"

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.alb_sg.id] # Reusing SG for simplicity in this demo
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              echo "Starting Inference Node"
              # Simulate a health check endpoint
              mkdir -p /var/www/html
              echo "OK" > /var/www/html/health
              # Start a simple python server
              cd /var/www/html
              python3 -m http.server 80 &
              EOF
  )

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "Sunnybrook Inference Node"
    }
  }
}

resource "aws_autoscaling_group" "inference_asg" {
  desired_capacity    = 2
  max_size            = 5
  min_size            = 1
  vpc_zone_identifier = [aws_subnet.public_subnet_a.id, aws_subnet.public_subnet_b.id]
  target_group_arns   = [aws_lb_target_group.inference_nodes.arn]

  launch_template {
    id      = aws_launch_template.inference_lt.id
    version = "$Latest"
  }

  tag {
    key                 = "Environment"
    value               = "Research"
    propagate_at_launch = true
  }
}
