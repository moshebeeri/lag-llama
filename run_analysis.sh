#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure we're in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "lag_llama" ]; then
    echo -e "${RED}Please activate the lag_llama conda environment first:${NC}"
    echo -e "${YELLOW}conda activate lag_llama${NC}"
    exit 1
fi

# Create necessary directories
mkdir -p data output models

# Run the analysis
echo -e "${YELLOW}Starting analysis...${NC}"
python src/models/tech_forecast.py

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Analysis completed successfully!${NC}"
    echo -e "\nResults saved in:"
    echo -e "${YELLOW}tech_forecasts.csv${NC}"
    echo -e "${YELLOW}tech_correlations.png${NC}"
else
    echo -e "${RED}Analysis failed. Please check the error messages above.${NC}"
    exit 1
fi 