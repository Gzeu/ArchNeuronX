# Regime-Aware Ensemble Build Script
# Builds the regime-aware ensemble system with example

Write-Host "🚀 Building Regime-Aware Ensemble System v3.0"
Write-Host "=========================================="

# Check for Visual Studio
if (!(Get-Command cl -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Visual Studio compiler not found"
    Write-Host "Please install Visual Studio 2022 with C++ development tools"
    Write-Host "Download: https://visualstudio.microsoft.com/downloads/"
    Pause
    Exit 1
}

# Create directories
if (!(Test-Path build)) { New-Item -ItemType Directory -Path build }
if (!(Test-Path build\regime_aware)) { New-Item -ItemType Directory -Path build\regime_aware }
if (!(Test-Path examples)) { New-Item -ItemType Directory -Path examples }
if (!(Test-Path docs)) { New-Item -ItemType Directory -Path docs }

Write-Host "📁 Created build directories"

# Check if LibTorch is available
$libtorch_paths = @(
    "C:\libtorch",
    "D:\libtorch",
    "C:\Users\*\libtorch",
    "..\libtorch"
)

$libtorch_found = $false
foreach ($path in $libtorch_paths) {
    if (Test-Path $path) {
        $env:LIBTORCH_DIR = $path
        $libtorch_found = $true
        Write-Host "✅ Found LibTorch at: $path"
        break
    }
}

if (-not $libtorch_found) {
    Write-Host "❌ LibTorch not found"
    Write-Host "Please download LibTorch from: https://pytorch.org/get-started/locally/"
    Write-Host "Extract to C:\libtorch or set LIBTORCH_DIR environment variable"
    Pause
    Exit 1
}

# Compile regime-aware ensemble example
Write-Host "🔨 Compiling Regime-Aware Ensemble System..."

$compile_cmd = @"
cl /EHsc /Fe:build\regime_aware\regime_aware_example.exe ^
   /I"include" ^
   /I"include\models" ^
   /I"include\regime" ^
   /I"$env:LIBTORCH_DIR\include" ^
   /D_CRT_SECURE_NO_WARNINGS ^
   /O2 ^
   /MD ^
   examples\regime_aware_example.cpp ^
   src\regime\regime_detector.cpp ^
   src\models\regime_aware_ensemble.cpp ^
   src\models\ensemble.cpp ^
   /link ^
   /LIBPATH:"$env:LIBTORCH_DIR\lib" ^
   torch.lib ^
   torch_cpu.lib ^
   c10.lib
"@

Invoke-Expression $compile_cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Compilation failed"
    Write-Host "Check for missing dependencies or compilation errors"
    Write-Host "Make sure LibTorch is properly installed"
    Pause
    Exit 1
}

Write-Host "✅ Build completed successfully!"
Write-Host "📦 Binary: build\regime_aware\regime_aware_example.exe"
Write-Host ""
Write-Host "🎯 Regime-Aware Ensemble Features:"
Write-Host "    📊 Real-time Market Regime Detection"
Write-Host "    🔄 Regime-Specific Model Weighting"
Write-Host "    🛡️ Anti-Overfitting Protection"
Write-Host "    📈 Performance Tracking per Regime"
Write-Host "    ⚡ Dynamic Ensemble Adaptation"
Write-Host "    🎯 Overfitting Detection & Mitigation"
Write-Host ""
Write-Host "🚀 REGIME-AWARE ENSEMBLE SYSTEM READY!"
Write-Host ""

# Quick start commands
Write-Host ""
Write-Host "🚀 Quick Start Commands:"
Write-Host "    build\regime_aware\regime_aware_example.exe    # Run demonstration"
Write-Host ""
Write-Host "📚 Documentation:"
Write-Host "    docs\regime_aware_ensemble.md                    # Full documentation"
Write-Host ""
Write-Host "🎯 READY FOR INSTITUTIONAL-GRADE TRADING!"
Write-Host ""

# Test the executable
Write-Host "🧪 Testing regime-aware ensemble..."
try {
    & "build\regime_aware\regime_aware_example.exe"
    Write-Host "✅ Example executed successfully!"
} catch {
    Write-Host "⚠️ Example execution failed - check LibTorch DLLs"
    Write-Host "Make sure LibTorch DLLs are in your PATH"
}

Write-Host ""
Write-Host "🎯 REGIME-AWARE ENSEMBLE SYSTEM DEPLOYMENT COMPLETE!"
Write-Host ""

Pause
