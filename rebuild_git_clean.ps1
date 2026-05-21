$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "Rebuilding a clean Git repository in:"
Write-Host $ProjectDir
Write-Host ""

if (Test-Path ".git") {
    Write-Host "Removing old inner .git folder..."
    Remove-Item -Recurse -Force ".git"
}

Write-Host "Creating new Git repository..."
git init

$localName = git config user.name
$localEmail = git config user.email

if ([string]::IsNullOrWhiteSpace($localName)) {
    git config user.name "ECG Project"
}

if ([string]::IsNullOrWhiteSpace($localEmail)) {
    git config user.email "ecg-project@example.local"
}

Write-Host "Adding project files..."
git add .

Write-Host ""
Write-Host "Files staged for the first clean commit:"
git status --short

Write-Host ""
Write-Host "Creating commit..."
git commit -m "rebuild clean ecg app repository"

Write-Host ""
Write-Host "Done. This folder now has a clean Git repository."
