# Terraform Configuration for CIBC Portfolio Optimizer on GCP
# Deploys a Google App Engine application

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "cibc-dividend-optimizer-v1"
  default     = "cibc-dividend-optimizer"
}

variable "region" {
  description = "GCP Region"
  default     = "northamerica-northeast1" # Montreal
}

# Enable App Engine API
resource "google_project_service" "app_engine" {
  project = var.project_id
  service = "appengine.googleapis.com"
  disable_on_destroy = false
}

# Create App Engine Application
resource "google_app_engine_application" "app" {
  project     = var.project_id
  location_id = var.region
  database_type = "CLOUD_FIRESTORE" # Use Firestore for state if needed
  
  depends_on = [google_project_service.app_engine]
}

# Cloud Build Trigger for CI/CD (Optional)
resource "google_cloudbuild_trigger" "deploy_trigger" {
  name = "deploy-portfolio-optimizer"
  project = var.project_id
  
  github {
    owner = "cibc-dev"
    name  = "portfolio-optimizer"
    push {
      branch = "^main$"
    }
  }
  
  filename = "cloudbuild.yaml"
}

output "url" {
  value = "https://${var.project_id}.appspot.com"
}
