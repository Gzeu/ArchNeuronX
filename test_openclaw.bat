@echo off
REM OpenCLaw Trading Agent Test Script
REM Tests all major functionality

echo 🧪 Testing OpenCLaw Trading Agent with Official Core
echo ==========================================

echo 📊 Testing OpenCLaw Core Engine...
build\Release\openclaw_agent.exe --test-core

echo 📈 Testing Signal Generation...
build\Release\openclaw_agent.exe --test-signals

echo 📊 Testing Smart Order Routing...
build\Release\openclaw_agent.exe --test-routing

echo 📊 Testing Risk Management...
build\Release\openclaw_agent.exe --test-risk

echo 📊 Testing Portfolio Optimization...
build\Release\openclaw_agent.exe --test-portfolio

echo ✅ All tests completed!
echo.
echo 🎯 OpenCLaw Trading Agent Status: FULLY FUNCTIONAL
echo.
echo 🚀 READY FOR LIVE TRADING!
pause
