$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
dotnet build native/ProcessLoopback/ProcessLoopback.csproj -c Release
if ($LASTEXITCODE -ne 0) { throw 'Process capture helper build failed' }
