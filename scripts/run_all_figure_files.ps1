# scripts/run_all_figure_files.ps1
# Run from the repository root in VSCode PowerShell after scripts/run_all_experiment_files.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts/run_all_figure_files.ps1
#
# This script generates figures from saved experiment outputs. It does not retrain models.
# It generates:
#   - trajectory-comparison plots for selected saved models
#   - NLR extrapolation heatmaps for lift and drag

param(
    [switch]$ContinueOnError,
    [int[]]$TrajectorySeeds = @(0, 1, 2, 3, 4),
    [int]$TrajectoryIndex = 0,
    [double[]]$XLim = $null,
    [string[]]$HeatmapModels = @("tdtf", "delay_mlp", "tddmd"),
    [switch]$AnnotateHeatmaps,
    [switch]$IncludeExtrapolationTrajectories,
    [switch]$Show
)

$ErrorActionPreference = "Stop"

$LogDir = "paper_run_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "run_all_figure_files_$Timestamp.log"

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

function Prediction-Path {
    param (
        [Parameter(Mandatory=$true)] [string]$Root,
        [Parameter(Mandatory=$true)] [string]$CaseName,
        [Parameter(Mandatory=$true)] [string]$ModelName,
        [Parameter(Mandatory=$true)] [int]$N,
        [AllowNull()] [Nullable[int]]$H,
        [Parameter(Mandatory=$true)] [int]$M,
        [Parameter(Mandatory=$true)] [int]$J,
        [AllowNull()] [Nullable[int]]$Seed
    )

    $HTag = if ($null -eq $H) { "none" } else { [string]$H }
    $SeedTag = if ($null -eq $Seed) { "none" } else { [string]$Seed }
    $FileName = "{0}_{1}_n={2}_h={3}_m={4}_J={5}_seed={6}.npz" -f $CaseName, $ModelName, $N, $HTag, $M, $J, $SeedTag
    return Join-Path (Join-Path $Root "predictions") $FileName
}

function Run-TrajectoryFigure {
    param (
        [Parameter(Mandatory=$true)] [string]$Name,
        [Parameter(Mandatory=$true)] [string]$Root,
        [Parameter(Mandatory=$true)] [string]$CaseName,
        [Parameter(Mandatory=$true)] [int]$N,
        [Parameter(Mandatory=$true)] [int]$H,
        [Parameter(Mandatory=$true)] [int]$J,
        [Parameter(Mandatory=$true)] [int]$Seed,
        [string[]]$Models = @("tddmd", "tdtf", "delay_mlp"),
        [double[]]$XLim = $null,
        [switch]$AllTrajectories
    )

    $PredictionFiles = New-Object System.Collections.Generic.List[string]

    foreach ($ModelName in $Models) {
        switch ($ModelName) {
            "tddmd" {
                $PredictionFiles.Add((Prediction-Path -Root $Root -CaseName $CaseName -ModelName "tddmd" -N $N -H $null -M 1 -J $J -Seed $null))
            }
            "tdtf" {
                $PredictionFiles.Add((Prediction-Path -Root $Root -CaseName $CaseName -ModelName "tdtf" -N $N -H $H -M 1 -J $J -Seed $Seed))
            }
            "delay_mlp" {
                $PredictionFiles.Add((Prediction-Path -Root $Root -CaseName $CaseName -ModelName "delay_mlp" -N $N -H $H -M 1 -J $J -Seed $Seed))
            }
            default {
                throw "Unknown trajectory model '$ModelName'. Valid options are: tddmd, tdtf, delay_mlp."
            }
        }
    }

    $ModelTag = ($Models -join "_")
    $SaveStem = Join-Path (Join-Path $Root "plots") ("{0}_compare_{1}_n={2}_h={3}_seed={4}" -f $CaseName, $ModelTag, $N, $H, $Seed)

    $PlotArgs = @(
        "-m", "scripts.plot_experiment_figures",
        "trajectory"
    )

    $PlotArgs += $PredictionFiles

    $PlotArgs += @(
        "--save-path", $SaveStem,
        "--traj-index", ([string]$TrajectoryIndex)
    )

    if ($null -ne $XLim) {
        if ($XLim.Count -ne 2) {
            throw "-XLim must contain exactly two values: xmin xmax."
        }

        $PlotArgs += @(
            "--xlim",
            ([string]$XLim[0]),
            ([string]$XLim[1])
        )
    }

    if ($AllTrajectories) {
        $PlotArgs += "--all-trajectories"
    }

    if ($Show) {
        $PlotArgs += "--show"
    }

    Run-Step $Name $PlotArgs
}

function Run-HeatmapFigure {
    param (
        [Parameter(Mandatory=$true)] [string]$Name,
        [Parameter(Mandatory=$true)] [string]$Root,
        [Parameter(Mandatory=$true)] [string]$Qoi,
        [Parameter(Mandatory=$true)] [string]$ModelName
    )

    $CsvPath = Join-Path (Join-Path $Root "results") ("{0}_agg_metrics.csv" -f $Root)

    $PlotArgs = @(
        "-m", "scripts.plot_experiment_figures",
        "heatmap",
        "--csv", $CsvPath,
        "--qoi", $Qoi,
        "--model", $ModelName,
        "--output-root", $Root
    )

    if ($AnnotateHeatmaps) {
        $PlotArgs += "--annotate"
    }

    if ($Show) {
        $PlotArgs += "--show"
    }

    Run-Step $Name $PlotArgs
}

try {
    foreach ($Seed in $TrajectorySeeds) {
        ######################### Sinus #######################

        Run-TrajectoryFigure `
            -Name "Sinus trajectory comparison, seed $Seed" `
            -Root "sinus" `
            -CaseName "sinus" `
            -N 2 `
            -H 10 `
            -J 50 `
            -Seed $Seed `
            -Models @("tddmd", "tdtf", "delay_mlp")

        # ######################### NLR lift / drag #######################

        Run-TrajectoryFigure `
            -Name "NLR lift trajectory comparison, seed $Seed" `
            -Root "nlr_lift" `
            -CaseName "nlr_lift" `
            -N 10 `
            -H 100 `
            -J 5000 `
            -Seed $Seed `
            -AllTrajectories

        Run-TrajectoryFigure `
            -Name "NLR drag trajectory comparison, seed $Seed" `
            -Root "nlr_drag" `
            -CaseName "nlr_drag" `
            -N 10 `
            -H 100 `
            -J 5000 `
            -Seed $Seed `
            -AllTrajectories

        if ($IncludeExtrapolationTrajectories) {
            Run-TrajectoryFigure `
                -Name "NLR lift extrapolation trajectory comparison, seed $Seed" `
                -Root "nlr_lift_extrapolation" `
                -CaseName "nlr_lift_extrapolation" `
                -N 10 `
                -H 100 `
                -J 5000 `
                -Seed $Seed `
                -AllTrajectories

            Run-TrajectoryFigure `
                -Name "NLR drag extrapolation trajectory comparison, seed $Seed" `
                -Root "nlr_drag_extrapolation" `
                -CaseName "nlr_drag_extrapolation" `
                -N 10 `
                -H 100 `
                -J 5000 `
                -Seed $Seed `
                -AllTrajectories
            }
    
        ######################### Lorenz63 #######################

        Run-TrajectoryFigure `
            -Name "Lorenz63 trajectory comparison n=1, seed $Seed" `
            -Root "lorenz63" `
            -CaseName "lorenz63" `
            -N 1 `
            -H 100 `
            -J 5000 `
            -Seed $Seed `
            -Models @("tddmd", "tdtf", "delay_mlp")


        Run-TrajectoryFigure `
            -Name "Lorenz63 trajectory comparison n=3, seed $Seed" `
            -Root "lorenz63" `
            -CaseName "lorenz63" `
            -N 3 `
            -H 100 `
            -J 5000 `
            -Seed $Seed `
            -Models @("tddmd", "tdtf", "delay_mlp") `
            -XLim @(49, 75)
    
        ######################### Reaction-Diffusion PDE #######################

        Run-TrajectoryFigure `
            -Name "Reaction-diffusion trajectory comparison, seed $Seed" `
            -Root "reacDiff" `
            -CaseName "reacDiff" `
            -N 10 `
            -H 100 `
            -J 2000 `
            -Seed $Seed `
            -Models @("tddmd", "tdtf", "delay_mlp")
    }
    

    ######################### NLR extrapolation heatmaps #######################

    foreach ($ModelName in $HeatmapModels) {
        Run-HeatmapFigure `
            -Name "NLR lift extrapolation heatmap, $ModelName" `
            -Root "nlr_lift_extrapolation" `
            -Qoi "lift" `
            -ModelName $ModelName

        Run-HeatmapFigure `
            -Name "NLR drag extrapolation heatmap, $ModelName" `
            -Root "nlr_drag_extrapolation" `
            -Qoi "drag" `
            -ModelName $ModelName
    }

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

    Write-Host "All figure files were generated successfully."
    Write-Host "Log file: $LogFile"
}
finally {
    Stop-Transcript
}
