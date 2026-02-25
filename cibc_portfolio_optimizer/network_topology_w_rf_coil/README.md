# Sunnybrook Imaging Research Cloud Architecture

![Architecture Diagram](sunnybrook_architecture_diagram_v2.png)

This project defines a scalable, cloud-native architecture designed to accelerate imaging research across four key specialties:
1.  **Neuroimaging**
2.  **Cardiovascular Imaging**
3.  **Pathology**
4.  **Mammography**

> **[Click here to view the detailed Implementation Walkthrough Plan](./IMPLEMENTATION_WALKTHROUGH.md)**

## Architecture Overview

The system utilizes a hybrid multi-cloud strategy leveraging **AWS**, **GCP**, and on-premise **Imaging Scanner Infrastructure**, with high-performance compute offloading to **NVIDIA Quantum** simulation environments where applicable.

### High-Level Components

1.  **Ingestion Layer**:
    *   Direct feeds from MRI/CT/Pathology Scanners via DICOM receivers.
    *   Edge gateways for immediate anonymization and buffering.
2.  **Data Lake & Storage**:
    *   **AWS S3** / **Google Cloud Storage** for raw DICOM/WSI (Whole Slide Imaging) storage.
    *   Lifecycle policies to move cold data to Glacier/Archive.
3.  **Compute & AI Acceleration**:
    *   **AWS SageMaker** for model training and deployment.
    *   **GCP Vertex AI** for specific pathology models (if preferred).
    *   **NVIDIA Quantum** (via CuQuantum/Qoda) for experimental quantum image reconstruction algorithms.
4.  **Orchestration**:
    *   **Kubernetes (EKS/GKE)** for managing containerized analysis pipelines.
    *   **Airflow** or **AWS Step Functions** for workflow orchestration.
5.  **Load Balancing**:
    *   **AWS Application Load Balancer (ALB)** to distribute inference requests across compute nodes.
    *   Health checks to ensure backend availability.

## Specialty-Specific Pipelines

### Neuroimaging
*   **Focus**: fMRI analysis, tractography.
*   **Acceleration**: GPU-accelerated massive parallel processing for voxel-wise analysis.

### Cardiovascular
*   **Focus**: 4D flow MRI, Calcium scoring.
*   **Acceleration**: Real-time segmentation of cardiac phases using deployed SageMaker endpoints.

### Pathology
*   **Focus**: Gigapixel WSI analysis.
*   **Acceleration**: Tiled processing using serverless functions (Lambda/Cloud Functions) to parallelize feature extraction.

### Mammography
*   **Focus**: Micro-calcification detection.
*   **Acceleration**: High-sensitivity Deep Learning models trained on large annotated datasets.

## Directory Structure

*   `neuroimaging/`: Specific analysis scripts and configs for Neuro.
*   `cardiovascular/`: Specific analysis scripts and configs for Cardio.
*   `pathology/`: Specific analysis scripts and configs for Pathology.
*   `mammography/`: Specific analysis scripts and configs for Mammography.
*   `cloud_infra/`: Terraform/IaC code to provision the environment.
*   `backend_integration/`: Python adaptors for NVIDIA Quantum, AWS, GCP, and Scanner protocols.
