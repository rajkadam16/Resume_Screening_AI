# Resume Screening AI - Project Restructuring Script
# Run this to reorganize your project structure

Write-Host "=== Resume Screening AI - Restructuring ===" -ForegroundColor Cyan
Write-Host ""

$root = "c:\RajKadam\khalsa mini project\Resume_Screening_AI"
$source = "$root\Dynamic_Resume_Screener"

# Create backup
Write-Host "[1/10] Creating backup..." -ForegroundColor Yellow
$backup = "$root`_BACKUP_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item $root $backup -Recurse -Force
Write-Host "Backup created: $backup" -ForegroundColor Green

# Create directories
Write-Host "[2/10] Creating directories..." -ForegroundColor Yellow
$dirs = @("app", "app\api", "app\core", "app\ml", "app\database", "app\utils", 
          "scripts", "scripts\setup", "scripts\data", "scripts\ml", "scripts\migration",
          "tests", "tools", "data", "data\resumes", "data\models", "data\exports", 
          "data\database", "logs", "docs", "archive", "archive\debug")

foreach ($d in $dirs) {
    $path = "$root\$d"
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# Create __init__.py files
$initDirs = @("app", "app\api", "app\core", "app\ml", "app\database", "app\utils", "tests")
foreach ($d in $initDirs) {
    $file = "$root\$d\__init__.py"
    if (-not (Test-Path $file)) {
        "" | Out-File -FilePath $file -Encoding UTF8
    }
}
Write-Host "Directories created" -ForegroundColor Green

# Move templates
Write-Host "[3/10] Moving templates..." -ForegroundColor Yellow
if (Test-Path "$source\templates") {
    Copy-Item "$source\templates\*" "$root\app\templates\" -Force -ErrorAction SilentlyContinue
}
Write-Host "Templates moved" -ForegroundColor Green

# Move core files
Write-Host "[4/10] Moving core files..." -ForegroundColor Yellow
$coreMap = @{
    "config.py" = "app\config.py"
    "database.py" = "app\database\sqlite.py"
    "mongodb_database.py" = "app\database\mongodb.py"
    "ml_models_improved.py" = "app\ml\models.py"
    "training_pipeline_improved.py" = "app\ml\training.py"
}

foreach ($key in $coreMap.Keys) {
    $src = "$source\$key"
    $dst = "$root\$($coreMap[$key])"
    if (Test-Path $src) {
        Copy-Item $src $dst -Force
    }
}
Write-Host "Core files moved" -ForegroundColor Green

# Move scripts
Write-Host "[5/10] Moving scripts..." -ForegroundColor Yellow
$scriptMap = @{
    "add_feedback_bulk.py" = "scripts\data\add_feedback_bulk.py"
    "add_feedback_quick.py" = "scripts\data\add_feedback_quick.py"
    "batch_resume_processor.py" = "scripts\data\batch_processor.py"
    "import_batch_results.py" = "scripts\data\import_batch_results.py"
    "migrate_to_mongodb.py" = "scripts\migration\migrate_to_mongodb.py"
    "train_improved_model.py" = "scripts\ml\train_model.py"
    "quick_start.py" = "scripts\setup\quick_start.py"
}

foreach ($key in $scriptMap.Keys) {
    $src = "$source\$key"
    $dst = "$root\$($scriptMap[$key])"
    if (Test-Path $src) {
        Copy-Item $src $dst -Force
    }
}
Write-Host "Scripts moved" -ForegroundColor Green

# Move tools
Write-Host "[6/10] Moving tools..." -ForegroundColor Yellow
$tools = @("check_db_status.py", "check_feedback_count.py", "check_feedback_distribution.py",
           "check_model_metadata.py", "check_training_result.py", "check_training_status.py", "show_data.py")

foreach ($t in $tools) {
    $src = "$source\$t"
    $dst = "$root\tools\$t"
    if (Test-Path $src) {
        Copy-Item $src $dst -Force
    }
}

if (Test-Path "$source\view_current_data.py") {
    Copy-Item "$source\view_current_data.py" "$root\tools\view_data.py" -Force
}
Write-Host "Tools moved" -ForegroundColor Green

# Archive debug files
Write-Host "[7/10] Archiving debug files..." -ForegroundColor Yellow
$debug = @("capture_error.py", "debug_data_format.py", "debug_training.py", "demo_scoring.py")
foreach ($d in $debug) {
    $src = "$source\$d"
    $dst = "$root\archive\debug\$d"
    if (Test-Path $src) {
        Copy-Item $src $dst -Force
    }
}
Write-Host "Debug files archived" -ForegroundColor Green

# Move data
Write-Host "[8/10] Moving data..." -ForegroundColor Yellow
if (Test-Path "$source\resumes") {
    Copy-Item "$source\resumes\*" "$root\data\resumes\" -Force -ErrorAction SilentlyContinue
}
if (Test-Path "$source\models") {
    Copy-Item "$source\models\*" "$root\data\models\" -Force -ErrorAction SilentlyContinue
}
if (Test-Path "$source\data") {
    Copy-Item "$source\data\*" "$root\data\database\" -Force -ErrorAction SilentlyContinue
}
Write-Host "Data moved" -ForegroundColor Green

# Move docs
Write-Host "[9/10] Moving documentation..." -ForegroundColor Yellow
if (Test-Path "$root\doc\README_FEEDING_RESUMES.md") {
    Copy-Item "$root\doc\README_FEEDING_RESUMES.md" "$root\docs\" -Force
}
Write-Host "Documentation moved" -ForegroundColor Green

# Create placeholders
Write-Host "[10/10] Creating placeholders..." -ForegroundColor Yellow
$keeps = @("data\resumes", "data\models", "data\exports", "logs")
foreach ($k in $keeps) {
    "" | Out-File -FilePath "$root\$k\.gitkeep" -Encoding UTF8 -Force
}
Write-Host "Placeholders created" -ForegroundColor Green

# Done
Write-Host ""
Write-Host "=== COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Backup: $backup" -ForegroundColor Cyan
Write-Host "Original: $source" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: Update imports in app.py and test!" -ForegroundColor Yellow
