#!/bin/bash

# Deploy Real Estate Dashboard to GitHub Repository
# Developed by Maksim Kitikov - Upside Analytics

echo "🚀 Deploying Real Estate Dashboard to GitHub Repository"
echo "Developer: Maksim Kitikov - Upside Analytics"
echo "=================================================="

# Configuration
REPO_URL="https://github.com/maksimkitikov/realestate1.git"
REPO_NAME="realestate1"
SSH_KEY_PATH="~/.ssh/github_realestate_deploy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if SSH key exists
if [ ! -f "$SSH_KEY_PATH" ]; then
    print_error "SSH key not found at $SSH_KEY_PATH"
    print_info "Generating new SSH key..."
    ssh-keygen -t ed25519 -C "maksimkitikov.uk@gmail.com" -f "$SSH_KEY_PATH" -N ""
    print_status "SSH key generated successfully"
fi

# Display SSH public key
echo ""
print_info "SSH Public Key for GitHub Deploy Keys:"
echo "=========================================="
cat "${SSH_KEY_PATH}.pub"
echo "=========================================="
echo ""

print_warning "IMPORTANT: Add this SSH key to GitHub repository:"
print_info "1. Go to: https://github.com/maksimkitikov/realestate1/settings/keys/new"
print_info "2. Title: Real Estate Dashboard Deploy Key"
print_info "3. Key: Copy the key above"
print_info "4. Check 'Allow write access'"
print_info "5. Click 'Add key'"
echo ""

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    print_info "Initializing git repository..."
    git init
    print_status "Git repository initialized"
fi

# Add remote origin if not exists
if ! git remote get-url origin > /dev/null 2>&1; then
    print_info "Adding GitHub remote origin..."
    git remote add origin "$REPO_URL"
    print_status "Remote origin added"
fi

# Configure git user
print_info "Configuring git user..."
git config user.name "Maksim Kitikov"
git config user.email "maksimkitikov.uk@gmail.com"
print_status "Git user configured"

# Add all files
print_info "Adding files to git..."
git add .
print_status "Files added to git"

# Commit changes
print_info "Committing changes..."
git commit -m "Deploy Real Estate Dashboard v2.0 - Maksim Kitikov - Upside Analytics

- Advanced US Real Estate Analytics Dashboard
- Interactive state-level analysis with regression models
- Real-time FRED API integration
- Machine learning models for market prediction
- Production-ready platform with R² scoring
- Developed by Maksim Kitikov - Upside Analytics"
print_status "Changes committed"

# Push to GitHub
print_info "Pushing to GitHub repository..."
git push -u origin main
print_status "Successfully pushed to GitHub"

echo ""
print_status "🎉 Deployment completed successfully!"
echo ""
print_info "Next steps for Render deployment:"
print_info "1. Go to https://render.com"
print_info "2. Create new Web Service"
print_info "3. Connect GitHub repository: maksimkitikov/realestate1"
print_info "4. Configure build settings:"
print_info "   - Build Command: pip install -r requirements.txt"
print_info "   - Start Command: python dashboard_advanced.py"
print_info "5. Add environment variables:"
print_info "   - DATABASE_URL: your_postgresql_connection_string"
print_info "6. Deploy!"
echo ""
print_info "Repository URL: https://github.com/maksimkitikov/realestate1"
print_info "Developer: Maksim Kitikov - Upside Analytics"
