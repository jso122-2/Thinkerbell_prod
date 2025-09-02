#!/bin/bash

echo "🎭 Thinkerbell Linux Setup Script"
echo "================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3 first."
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ Python version: $(python3 --version)"

# Install main dependencies
echo ""
echo "📦 Installing main project dependencies..."
npm install

# Install webapp dependencies
echo ""
echo "📦 Installing webapp dependencies..."
cd webapp
npm install

# Install Tailwind plugins
echo ""
echo "🎨 Installing Tailwind CSS plugins..."
npm install @tailwindcss/forms @tailwindcss/typography

# Go back to root
cd ..

# Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
pip3 install fastapi uvicorn sentence-transformers scikit-learn numpy

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the system:"
echo "1. Terminal 1: python3 backend_server.py"
echo "2. Terminal 2: npm run api:start"
echo "3. Terminal 3: cd webapp && npm start"
echo ""
echo "Or run all at once: npm run dev"
echo ""
echo "📡 Services will be available at:"
echo "- Python Backend: http://localhost:8000"
echo "- Node.js API: http://localhost:3000"
echo "- React Webapp: http://localhost:3001" 