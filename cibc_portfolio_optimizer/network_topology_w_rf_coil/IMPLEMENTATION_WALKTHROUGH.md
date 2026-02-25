# Sunnybrook Imaging Research Architecture - Implementation Walkthrough

![Console Preview](sunnybrook_console_preview.png)

This document outlines the step-by-step implementation plan for deploying the Sunnybrook Imaging Research cloud architecture. This walkthrough covers infrastructure provisioning, security hardening, management console deployment, and specialty-specific pipelines.

## Phase 1: Infrastructure Provisioning (Day 1-2)

**Goal**: Establish the secure cloud foundation on AWS.

1.  **Prerequisites**:
    *   Install [Terraform](https://www.terraform.io/downloads).
    *   Configure AWS CLI with credentials having admin permissions.
    *   Verify `cloud_infra/terraform/main.tf` region matches your compliance requirements (default: `ca-central-1`).

2.  **Initialize Terraform**:
    ```bash
    cd cloud_infra/terraform
    terraform init
    ```

3.  **Plan and Apply**:
    Review the resources to be created (S3 Buckets, IAM Roles, SageMaker Instances).
    ```bash
    terraform plan
    terraform apply
    ```
    *   **Action**: Confirm the apply with `yes`.
    *   **Output**: Note the S3 Bucket name, ECR Repository URL, and the **ALB DNS Name**.

4.  **Verification**:
    *   Log into the AWS Console.
    *   Confirm the S3 bucket exists and has "Block Public Access" enabled.
    *   Check that the SageMaker Notebook instance `sunnybrook-research-notebook` is "InService".
    *   Verify the **Application Load Balancer** is active and its target group is healthy.

## Phase 1.5: Security Hardening (Day 2)

**Goal**: Enable strict encryption and access controls.

1.  **KMS Encryption**:
    *   The updated Terraform now creates a customer-managed KMS Keys (`Sunnybrook-Imaging-KMS`).
    *   Confirm S3 buckets use correct Server-Side Encryption (SSE-KMS).
2.  **Network Lockdown**:
    *   Verify the `alb_sg` Security Group only accepts HTTP/HTTPS from authorized ranges (currently 0.0.0.0/0 for demo).
    *   Ensure Private Subnets (if added) route through NAT Gateways.

## Phase 2: Backend Integration Service (Day 3-5)

**Goal**: Connect local scanner streams to the cloud.

1.  **Environment Setup**:
    ```bash
    cd ../../backend_integration
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    *   Edit `scanner_integration_service.py` to set the real IPs of your DICOM receivers or modify the mock `ScannerInterface` for testing.
    *   Ensure GCP credentials are set (e.g., `GOOGLE_APPLICATION_CREDENTIALS`) if using Vertex AI.

3.  **Deploy Integration Service**:
    *   Ideally, this runs on a secure edge server or gateway (e.g., AWS Storage Gateway or a local Linux server).
    *   Run the service:
        ```bash
        python scanner_integration_service.py
        ```
    *   **Verification**: Check logs to see mock data flowing from "Scanner" -> "S3".

## Phase 2.5: Interactive Management Console (Day 5)

**Goal**: Launch the visual "Command Center" dashboard.

1.  **Launch**:
    ```bash
    cd sunnybrook_console/backend
    python app.py
    ```
2.  **Access**:
    *   Open `http://localhost:3000` in your browser.
3.  **Features to Test**:
    *   **Network Topology**: Drag nodes around in the D3.js visualization.
    *   **Security Controls**: Click "Rotate KMS Keys" to simulate key rotation.
    *   **Emergency**: Click **"INITIATE LOCKDOWN"** to trigger the red-alert UI state.

## Phase 3: Specialty Pipeline Deployment (Day 6-10)

**Goal**: Activate analysis modules for each department.

### 3.1 Neuroimaging
1.  Navigate to `neuroimaging/`.
2.  Deploy `fmri_analysis.py` as a SageMaker Processing Job or Lambda function.
3.  Test with a sample NIfTI file:
    ```python
    from neuroimaging.fmri_analysis import analyze_fmri_connectivity
    result = analyze_fmri_connectivity("patient_001", mock_data)
    print(result)
    ```

### 3.2 Cardiovascular
1.  Navigate to `cardiovascular/`.
2.  The `flow_4d_analysis.py` module computes Wall Shear Stress.
3.  Integrate this into the `scanner_integration_service.py` to trigger automatically upon receiving cardiac MRI tags.

### 3.3 Pathology & Mammography
1.  **Pathology**: Deploy `wsi_tiling.py` to a highly parallel environment (e.g., AWS Batch) because WSI files are massive.
2.  **Mammography**: `calcification_detection.py` is an inference script. Pack this into a Docker container:
    *   Create a `Dockerfile`.
    *   Build and push to the ECR repo created in Phase 1.
    *   Deploy as a SageMaker Endpoint for real-time inference.

## Phase 4: Quantum Acceleration (Experimental)

**Goal**: Test quantum algorithms for image reconstruction.

1.  **Setup**: Ensure you have a GPU-enabled environment (e.g., `g4dn` instance on AWS).
2.  **Library**: Install NVIDIA CuQuantum SDK.
3.  **Integration**:
    *   The `NvidiaQuantumSimulator` class in `backend_integration/scanner_integration_service.py` serves as the entry point.
    *   Run benchmarks comparing Classical FFT vs. Quantum FFT for image reconstruction.
    *   **Quantum**: Verify `Sunnybrook-Quantum-Sim` job submission.

## Phase 4.5: Cost Optimization & FinOps (Day 9)

**Goal**: Enable real-time financial monitoring and monetization.

1.  **Features**:
    *   **Spot Instance Orchestration**: 85% of compute loads effectively moved to Spot instances.
    *   **Departmental Chargeback**: Automated tracking of usage by specialty (Neuro, Cardio, etc.).
    *   **Dashboard**: Use the "Cost & Billing" tab to view estimated monthly bills and savings.
2.  **Verification**:
    *   Open Console -> "Cost & Billing".
    *   Verify "Spot Savings" is positive (green).

## Phase 5: Monitoring & Compliance (Day 11+)

1.  **CloudWatch**: Set up alarms for S3 bucket size and SageMaker error rates.
2.  **Audit Logs**: Enable CloudTrail to track who accesses patient data.
3.  **Compliance Review**: Ensure "Glacier" lifecycle rules in Terraform are active to meet data retention policies for Sunnybrook.

---
**Next Steps**:
*   [ ] Validate Terraform apply in a sandbox account.
*   [ ] Gather sample anonymized datasets for each specialty.
*   [ ] Schedule a review with the IT Security team.
