# Builds a self-contained copy of the application in dist\Granular Music Visualizer\
# (PyInstaller; no Python needed to run it). Models are copied from .model-cache and are
# local files: the build folder is not committed and must not be redistributed while it
# holds weights derived from non-commercial data.
#
#   .\"Build App.ps1"                                  # no guitar recognition model
#   .\"Build App.ps1" -GuitarModel guitar_mn10_v1      # start with this model switched on
#   .\"Build App.ps1" -GuitarModel guitar_mn10_v1 -AlsoInclude guitar_mn10_rv1
#                                  # further models copied in, to choose in the settings drawer
param([string]$GuitarModel = "", [string[]]$AlsoInclude = @())
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$name = 'Granular Music Visualizer'
if (Get-Process -Name $name -ErrorAction SilentlyContinue) { throw "Close $name before building: the build replaces its folder" }
$helper = 'native/ProcessLoopback/bin/Release/net9.0-windows'
if (-not (Test-Path "$helper/ProcessLoopback.exe")) { & "$PSScriptRoot/Build Process Capture.ps1" }
# A private .NET runtime for the helper (official build, checked against Microsoft's
# published SHA-512), so the application runs whatever .NET the machine has.
$runtime = "$helper/dotnet"
if (-not (Test-Path "$runtime/host/fxr")) {
    $version = '9.0.23'
    $sha512 = '9d8a933589fa79a381d1b831f07ef79793de3f406262bc076b969ea01a85744d6bb45dd64458ee8949e0d69574fb0f0b8949eb3b939afcecf9d071f9615c6acf'
    $zip = Join-Path $env:TEMP "dotnet-runtime-$version-win-x64.zip"
    Invoke-WebRequest "https://builds.dotnet.microsoft.com/dotnet/Runtime/$version/dotnet-runtime-$version-win-x64.zip" -OutFile $zip -UseBasicParsing
    if ((Get-FileHash $zip -Algorithm SHA512).Hash -ne $sha512) { throw 'The .NET runtime download does not match its checksum' }
    Expand-Archive $zip $runtime -Force
    Remove-Item $zip
}
if (-not (Test-Path '.venv/Scripts/python.exe')) { throw 'Run Setup.ps1 first' }
& .venv/Scripts/python.exe -c "import PyInstaller, onnxruntime" 2>$null
if ($LASTEXITCODE -ne 0) {
    uv pip install --python .venv/Scripts/python.exe --quiet "pyinstaller>=6.16" "onnxruntime>=1.23,<2"
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the build tools' }
}
& .venv/Scripts/python.exe -m PyInstaller --noconfirm --clean --windowed --name $name `
    --paths src --distpath dist --workpath dist/.work --specpath dist/.work `
    --collect-submodules gmv.recognition --hidden-import onnxruntime `
    --exclude-module pytest --exclude-module mir_eval --exclude-module matplotlib `
    packaging/app.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }
$out = "dist/$name"
Copy-Item $helper "$out/ProcessLoopback" -Recurse -Force
New-Item -ItemType Directory -Force "$out/models" | Out-Null
$defaults = @{}
if (Test-Path '.model-cache/activity_v2.npz') { Copy-Item '.model-cache/activity_v2.npz' "$out/models" }
foreach ($model in @($GuitarModel) + $AlsoInclude | Where-Object { $_ }) {
    foreach ($suffix in '.json', '.onnx', '.frontend.npz') {
        Copy-Item ".model-cache/$model$suffix" "$out/models"
    }
}
if ($GuitarModel) { $defaults['guitar_model'] = "$GuitarModel.json" }
$defaults | ConvertTo-Json | Set-Content "$out/models/defaults.json" -Encoding utf8
Write-Host "Built $out\$name.exe"
