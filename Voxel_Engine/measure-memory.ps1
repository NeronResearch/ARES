# measure-memory.ps1
param(
  [string]$ExePath = ".\bin\main.exe",
  [string]$Arguments = "E:\Code\Neron\ARES\Scenarios\FernBellPark\scenario.json 500 525",
  [int]$SampleMs = 100,
  [string]$OutCsv = ".\mem_samples.csv"
)

# Start the process and get a Process object
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $ExePath
$startInfo.Arguments = $Arguments
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardOutput = $false
$startInfo.RedirectStandardError = $false

$proc = [System.Diagnostics.Process]::Start($startInfo)

if (-not $proc) {
  Write-Error "Failed to start process $ExePath"
  exit 1
}

# Prepare CSV
"Timestamp,Pid,WorkingSetBytes,PrivateBytes" | Out-File -FilePath $OutCsv -Encoding utf8

$peakWorking = 0
$peakPrivate = 0

try {
  while (-not $proc.HasExited) {
    $proc.Refresh()
    $ws = $proc.WorkingSet64
    $pb = $proc.PrivateMemorySize64

    # update peaks
    if ($ws -gt $peakWorking) { $peakWorking = $ws }
    if ($pb -gt $peakPrivate) { $peakPrivate = $pb }

    # write sample
    ("{0},{1},{2},{3}" -f (Get-Date -Format o), $proc.Id, $ws, $pb) | Out-File -FilePath $OutCsv -Append -Encoding utf8

    Start-Sleep -Milliseconds $SampleMs
  }

  # final refresh after exit (capture final numbers)
  $proc.Refresh()
  $ws = $proc.WorkingSet64
  $pb = $proc.PrivateMemorySize64
  if ($ws -gt $peakWorking) { $peakWorking = $ws }
  if ($pb -gt $peakPrivate) { $peakPrivate = $pb }
  ("{0},{1},{2},{3}" -f (Get-Date -Format o), $proc.Id, $ws, $pb) | Out-File -FilePath $OutCsv -Append -Encoding utf8

} finally {
  # Report results
  function ToHuman($bytes) {
    if ($bytes -ge 1TB) { return "{0:N2} TB" -f ($bytes/1TB) }
    if ($bytes -ge 1GB) { return "{0:N2} GB" -f ($bytes/1GB) }
    if ($bytes -ge 1MB) { return "{0:N2} MB" -f ($bytes/1MB) }
    if ($bytes -ge 1KB) { return "{0:N2} KB" -f ($bytes/1KB) }
    return "$bytes bytes"
  }

  Write-Output ""
  Write-Output "Process Id: $($proc.Id)"
  Write-Output "Peak Working Set (physical): $peakWorking bytes (`"$(ToHuman $peakWorking)`")"
  Write-Output "Peak Private Bytes (committed): $peakPrivate bytes (`"$(ToHuman $peakPrivate)`")"
  Write-Output "Sample log saved to: $OutCsv"
}
