#!/bin/bash
# ═══════════════════════════════════════════════════════
#  EmotiChat — START
# ═══════════════════════════════════════════════════════

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$PROJECT_DIR/backend/venv" ]; then
    echo "ERROR: Run 'bash setup.sh' first."
    exit 1
fi

# ENV is loaded by python-dotenv in the app, not by shell

echo ""
echo "═══════════════════════════════════════════"
echo "  EmotiChat — Starting at http://localhost:8502"
echo "  Press Ctrl+C to stop"
echo "═══════════════════════════════════════════"
echo ""

cd "$PROJECT_DIR/backend"
source venv/bin/activate
python3 -m streamlit run streamlit_app.py \
    --server.port 8502 \
    --server.headless false \
    --browser.gatherUsageStats false
