# scripts/run_all_experiment_files.ps1
# Run from the repository root in VSCode PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts/run_all_experiment_files.ps1
#
# This script only runs experiments and saves files. It does not call plotting.
# It runs:
#   - TD-TF
#   - Delay-MLP
#   - TD-DMD baseline, produced automatically by scripts.run_experiment
# for seeds 0 1 2 3 4.

param(
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

$LogDir = "paper_run_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "run_all_experiment_files_$Timestamp.log"

Start-Transcript -Path $LogFile

$Failures = New-Object System.Collections.Generic.List[string]

function Run-Step {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Name,

        [Parameter(Mandatory=$true)]
        [string[]]$Args
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Name
    Write-Host "============================================================"
    Write-Host "py $($Args -join ' ')"

    & py @Args
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        $Message = "Step failed with exit code ${ExitCode}: ${Name}"
        Write-Host $Message
        $Failures.Add($Message)

        if (-not $ContinueOnError) {
            throw $Message
        }
    }
}

try {
    ######################### Sinus #######################

    Run-Step "Sinus experiment files" @(
        "-m", "scripts.run_experiment",
        "--case", "sinus",
        "--models", "tdtf", "delay_mlp",
        "--n", "2",
        "--h", "10",
        "--seeds", "0", "1", "2", "3", "4",
        "--J", "50",
        "--batch-size", "5",
        "--epochs", "1000",
        "--lr", "1e-2",
        "--save-predictions"
    )


    ######################### NLR lift #######################

    Run-Step "NLR lift experiment files" @(
        "-m", "scripts.run_experiment",
        "--case", "nlr",
        "--qoi", "lift",
        "--models", "tdtf", "delay_mlp",
        "--tau", "30",
        "--n", "10",
        "--h", "100",
        "--seeds", "0", "1", "2", "3", "4",
        "--J", "5000",
        "--batch-size", "100",
        "--epochs", "500",
        "--lr", "1e-2",
        "--replace",
        "--save-predictions"
    )


    ######################### NLR drag #######################

    Run-Step "NLR drag experiment files" @(
        "-m", "scripts.run_experiment",
        "--case", "nlr",
        "--qoi", "drag",
        "--models", "tdtf", "delay_mlp",
        "--tau", "30",
        "--n", "10",
        "--h", "100",
        "--seeds", "0", "1", "2", "3", "4",
        "--J", "5000",
        "--batch-size", "100",
        "--epochs", "500",
        "--lr", "1e-2",
        "--replace",
        "--save-predictions"
    )

    ######################### lift (extrapolation) #######################

    Run-Step "NLR lift experiment files" @(
        "-m", "scripts.run_experiment",
        "--case", "nlr",
        "--nlr-variant", "extrapolation",
        "--qoi", "lift",
        "--models", "tdtf", "delay_mlp",
        "--tau", "30",
        "--n", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
        "--h", "50", "100", "150",
        "--seeds", "0", "1", "2", "3", "4",
        "--J", "5000",
        "--batch-size", "100",
        "--epochs", "500",
        "--lr", "1e-2",
        "--weight-decay", "1e-2",
        "--replace",
        "--save-predictions"
    )


    ######################### drag (extrapolation) #######################

    Run-Step "NLR drag experiment files" @(
        "-m", "scripts.run_experiment",
        "--case", "nlr",
        "--nlr-variant", "extrapolation",
        "--qoi", "drag",
        "--models", "tdtf", "delay_mlp",
        "--tau", "30",
        "--n", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
        "--h", "50", "100", "150",
        "--seeds", "0", "1", "2", "3", "4",
        "--J", "5000",
        "--batch-size", "100",
        "--epochs", "500",
        "--lr", "1e-2",
        "--weight-decay", "1e-2",
        "--replace",
        "--save-predictions"
    )


    # ######################### Lorenz63 #######################

    Run-Step "Lorenz63 experiment files" @(
        "-m", "scripts.run_experiment",
        "--case", "lorenz63",
        "--models", "tdtf", "delay_mlp",
        "--tau", "16",
        "--n", "1", "3",
        "--h", "100",
        "--seeds", "0", "1", "2", "3", "4",
        "--J", "5000",
        "--batch-size", "100",
        "--epochs", "500",
        "--lr", "1e-2",
        # "--weight-decay", "1e-5",
        "--save-predictions"
    )


    # ######################### Reaction-Diffusion PDE #######################

    Run-Step "Reaction-diffusion experiment files" @(
        "-m", "scripts.run_experiment",
        "--case", "reacdiff",
        "--models", "tdtf", "delay_mlp",
        "--tau", "5",
        "--n", "10",
        "--h", "100",
        "--seeds", "0", "1", "2", "3", "4",
        "--J", "2000",
        "--m", "1",
        "--batch-size", "100",
        "--epochs", "500",
        "--lr", "1e-2",
        "--weight-decay", "1e-5",
        "--save-predictions"
    )


    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Run complete"
    Write-Host "============================================================"

    if ($Failures.Count -gt 0) {
        Write-Host "Completed with failures:"
        foreach ($Failure in $Failures) {
            Write-Host "  - $Failure"
        }
        exit 1
    }

    Write-Host "All experiment files were generated successfully."
    Write-Host "Log file: $LogFile"
}
finally {
    Stop-Transcript
}
