#!/bin/bash
# ═══════════════════════════════════════════════════════
#  EmotiChat — ONE-TIME SETUP
# ═══════════════════════════════════════════════════════
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "═══════════════════════════════════════════"
echo "  EmotiChat — Setting Up"
echo "═══════════════════════════════════════════"
echo ""

# ── Check Python 3.11 ──
echo "[1/3] Checking Python 3.11..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif python3 --version 2>&1 | grep -q "3.11"; then
    PYTHON_CMD="python3"
else
    echo "ERROR: Python 3.11 not found. Install: brew install python@3.11"
    exit 1
fi
echo "  Found: $($PYTHON_CMD --version)"

# ── Check .env ──
echo "[2/3] Checking .env..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "  Created .env — please set your GROQ_API_KEY in it"
fi

# ── Install packages ──
echo "[3/3] Installing Python packages..."
cd "$PROJECT_DIR/backend"

if [ -d "venv" ]; then
    echo "  Existing venv found, using it..."
else
    $PYTHON_CMD -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "  Done."

# Verify
python3 -c "
import fastapi, torch, transformers, librosa, cv2, groq, streamlit
print('  All packages verified OK')
"
deactivate

echo ""
echo "═══════════════════════════════════════════"
echo "  SETUP COMPLETE!"
echo ""
echo "  Run:  bash start.sh"
echo "═══════════════════════════════════════════"
echo ""
