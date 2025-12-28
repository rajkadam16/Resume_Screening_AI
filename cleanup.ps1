# Resume Screening AI - Cleanup Script
# Removes unnecessary files after restructuring

Write-Host "=== Cleaning Up Unnecessary Files ===" -ForegroundColor Cyan
Write-Host ""

$root = "c:\RajKadam\khalsa mini project\Resume_Screening_AI"
$source = "$root\Dynamic_Resume_Screener"

# Files to delete
Write-Host "[1/5] Removing temporary files..." -ForegroundColor Yellow

$tempFiles = @(
    "$source\debug_out.txt",
    "$source\training_error.log",
    "$source\training_output.txt",
    "$source\verify_output.txt",
    "$source\batch_results.json"  # 4.5MB file
)

$deletedCount = 0
foreach ($file in $tempFiles) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  Deleted: $(Split-Path $file -Leaf)" -ForegroundColor Gray
        $deletedCount++
    }
}
Write-Host "Removed $deletedCount temporary files" -ForegroundColor Green

# Remove obsolete fix scripts
Write-Host "[2/5] Removing obsolete fix scripts..." -ForegroundColor Yellow

$obsoleteScripts = @(
    "$source\fix_line_951.py",
    "$source\fix_results.py",
    "$source\verify_fix.py",
    "$source\resume_data.py"  # Only 151 bytes, likely empty
)

$deletedCount = 0
foreach ($file in $obsoleteScripts) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  Deleted: $(Split-Path $file -Leaf)" -ForegroundColor Gray
        $deletedCount++
    }
}
Write-Host "Removed $deletedCount obsolete scripts" -ForegroundColor Green

# Remove old/duplicate files (already moved to new structure)
Write-Host "[3/5] Removing duplicate files..." -ForegroundColor Yellow

$duplicates = @(
    "$source\ml_models.py",  # We kept ml_models_improved.py
    "$source\training_pipeline.py"  # We kept training_pipeline_improved.py
)

$deletedCount = 0
foreach ($file in $duplicates) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  Deleted: $(Split-Path $file -Leaf)" -ForegroundColor Gray
        $deletedCount++
    }
}
Write-Host "Removed $deletedCount duplicate files" -ForegroundColor Green

# Remove __pycache__
Write-Host "[4/5] Removing Python cache..." -ForegroundColor Yellow

$cacheCount = 0
Get-ChildItem -Path $root -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Remove-Item $_.FullName -Recurse -Force
    Write-Host "  Deleted: $($_.FullName)" -ForegroundColor Gray
    $cacheCount++
}
Write-Host "Removed $cacheCount cache directories" -ForegroundColor Green

# Optional: Remove entire Dynamic_Resume_Screener folder
Write-Host "[5/5] Dynamic_Resume_Screener folder..." -ForegroundColor Yellow
Write-Host ""
Write-Host "The old Dynamic_Resume_Screener folder still exists." -ForegroundColor White
Write-Host "It contains app.py and other files that may still be in use." -ForegroundColor White
Write-Host ""
Write-Host "Options:" -ForegroundColor Cyan
Write-Host "  1. Keep it (safe, recommended until you test)" -ForegroundColor White
Write-Host "  2. Delete it (only if new structure is working)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Delete Dynamic_Resume_Screener folder? (y/N)"

if ($choice -eq 'y' -or $choice -eq 'Y') {
    Write-Host ""
    Write-Host "WARNING: This will delete the entire folder!" -ForegroundColor Red
    Write-Host "Make sure you have:" -ForegroundColor Yellow
    Write-Host "  - Moved app.py to app/main.py" -ForegroundColor White
    Write-Host "  - Updated all imports" -ForegroundColor White
    Write-Host "  - Tested the application" -ForegroundColor White
    Write-Host ""
    
    $confirm = Read-Host "Are you SURE? Type 'DELETE' to confirm"
    
    if ($confirm -eq 'DELETE') {
        Remove-Item $source -Recurse -Force
        Write-Host "Deleted: Dynamic_Resume_Screener/" -ForegroundColor Green
    } else {
        Write-Host "Cancelled - folder kept" -ForegroundColor Yellow
    }
} else {
    Write-Host "Kept: Dynamic_Resume_Screener/" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "=== Cleanup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Cleaned:" -ForegroundColor Cyan
Write-Host "  - Temporary files (.txt, .log, .json)" -ForegroundColor White
Write-Host "  - Obsolete fix scripts" -ForegroundColor White
Write-Host "  - Duplicate/old files" -ForegroundColor White
Write-Host "  - Python cache (__pycache__)" -ForegroundColor White
Write-Host ""
Write-Host "Your project is now clean and organized!" -ForegroundColor Green
Write-Host ""
