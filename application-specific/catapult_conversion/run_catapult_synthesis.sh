#!/bin/bash
# Catapult AI NN Synthesis Script for EC2
# Run this script on your EC2 instance after copying the project

set -e

# Configuration
CATAPULT_HOME="/data/tools/catapult/Mgc_home"
export PATH="$CATAPULT_HOME/bin:$PATH"
export MGLS_LICENSE_FILE="29000@10.9.8.8"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Catapult AI NN Synthesis Runner      ${NC}"
echo -e "${GREEN}========================================${NC}"

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 <project_directory> [options]${NC}"
    echo ""
    echo "Options:"
    echo "  --synth-only    Run synthesis only (no simulation)"
    echo "  --rtl           Generate RTL output"
    echo "  --report        Generate reports only"
    echo ""
    echo "Example:"
    echo "  $0 ~/catapult_projects/my_cnn_project"
    exit 1
fi

PROJECT_DIR="$1"
shift

# Parse options
SYNTH_ONLY=false
GEN_RTL=false
REPORT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --synth-only)
            SYNTH_ONLY=true
            shift
            ;;
        --rtl)
            GEN_RTL=true
            shift
            ;;
        --report)
            REPORT_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Verify project directory
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Error: Project directory not found: $PROJECT_DIR${NC}"
    exit 1
fi

# Check for required files
echo -e "\n${YELLOW}Checking project files...${NC}"

REQUIRED_FILES=("build_prj.tcl" "myproject.cpp")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$PROJECT_DIR/$file" ]; then
        echo -e "${RED}Error: Missing required file: $file${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} $file"
done

# Check Catapult installation
echo -e "\n${YELLOW}Checking Catapult installation...${NC}"

if [ ! -f "$CATAPULT_HOME/bin/catapult" ]; then
    echo -e "${RED}Error: Catapult not found at $CATAPULT_HOME${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Catapult found"

# Check license
echo -e "\n${YELLOW}Checking license server...${NC}"
if ! timeout 5 bash -c "echo >/dev/tcp/10.9.8.8/29000" 2>/dev/null; then
    echo -e "${RED}Warning: Cannot reach license server at 10.9.8.8:29000${NC}"
    echo -e "${YELLOW}Continuing anyway...${NC}"
else
    echo -e "  ${GREEN}✓${NC} License server reachable"
fi

# Create output directory
OUTPUT_DIR="$PROJECT_DIR/catapult_output"
mkdir -p "$OUTPUT_DIR"

# Change to project directory
cd "$PROJECT_DIR"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Starting Catapult Synthesis          ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  Project: $PROJECT_DIR"
echo -e "  Output:  $OUTPUT_DIR"
echo ""

# Run Catapult
if [ "$REPORT_ONLY" = true ]; then
    echo -e "${YELLOW}Running report generation only...${NC}"
    catapult -shell -f build_prj.tcl 2>&1 | tee "$OUTPUT_DIR/catapult_log.txt"
else
    echo -e "${YELLOW}Running full synthesis...${NC}"
    catapult -shell -f build_prj.tcl 2>&1 | tee "$OUTPUT_DIR/catapult_log.txt"
fi

# Check for errors
if grep -q "Error:" "$OUTPUT_DIR/catapult_log.txt"; then
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}  Synthesis completed with errors      ${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check the log file: $OUTPUT_DIR/catapult_log.txt"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Synthesis Complete!                  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Output files: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "  - catapult_log.txt     : Full synthesis log"
echo "  - Catapult/            : Catapult project directory"

# List generated files
if [ -d "Catapult" ]; then
    echo ""
    echo "Generated RTL files:"
    find Catapult -name "*.v" -o -name "*.vhd" 2>/dev/null | head -10
fi

echo ""
echo -e "${GREEN}Done!${NC}"
