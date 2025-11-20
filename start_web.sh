#!/bin/bash

# Mudra Detection Web Application Startup Script

echo "=============================================="
echo "🙏 MUDRA DETECTION WEB APPLICATION 🙏"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "📥 Checking dependencies..."
pip install -q -r requirements.txt

# Check if camera is available
echo "📹 Checking camera..."
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('✅ Camera detected' if cap.isOpened() else '❌ Camera not found'); cap.release()"

echo ""
echo "=============================================="
echo "🚀 Starting Flask server..."
echo "=============================================="
echo ""
echo "📡 Access the application at:"
echo "   Local:   http://localhost:5000"
echo "   Network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo ""
echo "=============================================="
echo ""

# Run the Flask app
python3 app.py
