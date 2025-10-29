#!/bin/bash

# Setup script for development environment
#
# REQUIREMENTS:
# - AWS CLI installed and configured with credentials that have access to CodeArtifact
# - Python 3.10 or higher installed

# Check if AWS CLI is installed
# Ask user for the origin .env file
read -p "Enter the path to your origin .env file (default: .env.example): " ORIGIN_ENV_FILE
ORIGIN_ENV_FILE=${ORIGIN_ENV_FILE:-.env.example}
if [ ! -f "$ORIGIN_ENV_FILE" ]; then
    echo "❌ The specified origin .env file '$ORIGIN_ENV_FILE' does not exist."
    echo "Aborting setup."
    exit 1
fi
# Copy the .env file
cp "$ORIGIN_ENV_FILE" .env
echo "Copied '$ORIGIN_ENV_FILE' to .env"

if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Access to private packages is required."
    echo "Please install the AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    echo "Aborting setup."
    exit 1
fi

echo "Logging in to AWS CodeArtifact..."
if ! aws codeartifact login --tool pip --repository private-pypi --domain invofox-private-domain --domain-owner 834847960948 --region eu-west-1; then
    echo "❌ AWS CodeArtifact login failed. You must have proper AWS credentials configured."
    echo "The required packages cannot be properly installed without access to the private repository."
    echo "For more information, see: https://docs.aws.amazon.com/codeartifact/latest/ug/tokens-authentication.html"
    echo "Aborting setup."
    exit 1
fi
export CODEARTIFACT_AUTH_TOKEN=`aws codeartifact get-authorization-token --domain invofox-private-domain --domain-owner 834847960948 --region eu-west-1 --query authorizationToken --output text`

echo "Creating pdm lock files..."
pdm run lock

# Install project and development dependencies
echo "Installing project and development dependencies..."
rm -rf .venv
pdm install -L pdm.dev.lock
if [ $? -ne 0 ]; then
    echo "❌ Failed to install project dependencies."
    echo "Aborting setup."
    exit 1
fi

# Ensure pre-commit hooks are installed
echo "Installing pre-commit hooks..."
pdm run pre-commit install


echo "Development environment setup complete!"
echo "⚠️Remember to install the recommended VS Code extensions for the best experience."
echo ""
echo "Your development environment is now configured with:"
echo "✅ PDM environment (.venv)"
echo "✅ AWS CodeArtifact for private packages"
echo "✅ Project dependencies installed in development mode"
echo "✅ Development tools (ruff, mypy, pytest, dvc, pre-commit, etc.)"
