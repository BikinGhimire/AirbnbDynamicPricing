#!/usr/bin/env pwsh

$branch = git rev-parse --abbrev-ref HEAD

if ($branch -eq "test" -or $branch -eq "release" or $branch -eq "dev") {
    Write-Host "Setting up .gitignore for branch $branch"

    $ignoreContent = @"
# Ignore Jupyter notebook files
*.ipynb

# Ignore HTML files
*.html

# Ignore __pycache__, .idea, and .ipynb_checkpoints directories
__pycache__/
.idea/
.ipynb_checkpoints/
"@

    Set-Content -Path .gitignore -Value $ignoreContent

    git add .gitignore
    git commit -m "Update .gitignore for $branch branch" --no-verify
}
