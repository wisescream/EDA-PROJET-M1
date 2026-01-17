# Setup script for Sentiment Analysis Project
# This script will set up the environment and download necessary models

Write-Host "🚀 Setting up Sentiment Analysis Environment..." -ForegroundColor Green

# Check if virtual environment exists
if (-Not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing Python packages..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Download SpaCy model
Write-Host "Downloading SpaCy French model..." -ForegroundColor Yellow
python -m spacy download fr_core_news_sm

# Install Jupyter kernel
Write-Host "Installing Jupyter kernel..." -ForegroundColor Yellow
python -m ipykernel install --user --name=eda_venv --display-name="Python (EDA)"

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start working:" -ForegroundColor Cyan
Write-Host "  1. Run: jupyter notebook" -ForegroundColor White
Write-Host "  2. Open: sentiment_analysis.ipynb" -ForegroundColor White
Write-Host "  3. Select kernel: Python (EDA)" -ForegroundColor White
Write-Host ""
