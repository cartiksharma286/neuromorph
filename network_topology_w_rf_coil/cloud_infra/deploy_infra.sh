#!/bin/bash
set -e

# sunnybrook-deploy.sh
# Usage: ./sunnybrook-deploy.sh [deploy|destroy] [scale_size]

COMMAND=$1
SCALE=${2:-2} # Default to 2 instances if not specified

TERRAFORM_DIR="terraform"

if [ -z "$COMMAND" ]; then
    echo "Usage: ./sunnybrook-deploy.sh [deploy|destroy] <scale_size>"
    exit 1
fi

echo "--- Sunnybrook Cloud Infrastructure Manager ---"
echo "Command: $COMMAND"
echo "Scale: $SCALE"

cd $TERRAFORM_DIR

if [ "$COMMAND" = "deploy" ]; then
    echo "Initializing Terraform..."
    terraform init

    # In a real scenario, we might override variables for ASG size here
    # For now, we will use sed/logic or just rely on the main.tf defaults unless we use vars
    # A robust way is passing -var "desired_capacity=$SCALE" if we defined variables.
    
    echo "Applying Infrastructure..."
    terraform apply -auto-approve

    echo "Fetching Outputs..."
    # Assuming we added outputs, we could grab the ALB DNS here.
    # For demonstration, we'll try to get the ALB DNS if the output was defined.
    # terraform output alb_dns_name
    
    echo "Deployment Complete."
    
elif [ "$COMMAND" = "destroy" ]; then
    echo "Destroying Infrastructure..."
    terraform destroy -auto-approve
    echo "Destruction Complete."
    
else
    echo "Invalid command. Use deploy or destroy."
    exit 1
fi
