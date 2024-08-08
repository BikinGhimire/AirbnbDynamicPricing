#!/usr/bin/env pwsh

Copy-Item -Path .\hooks\post-checkout.ps1 -Destination .git\hooks\post-checkout.ps1
Copy-Item -Path .\hooks\post-checkout.bat -Destination .git\hooks\post-checkout.bat

Write-Host "Hooks installed successfully."
