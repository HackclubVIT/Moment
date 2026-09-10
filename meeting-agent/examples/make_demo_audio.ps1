# Synthesize a demo meeting WAV from the sample transcript using Windows SAPI voices.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File make_demo_audio.ps1 -Out demo_meeting.wav
param(
    [string]$Out = "demo_meeting.wav"
)

Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0

# Try to vary voices per speaker for a more realistic demo.
$voices = @($synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name })
$speakerVoice = @{}
$i = 0

try {
    $synth.SetOutputToWaveFile($Out)
    $lines = Get-Content (Join-Path $PSScriptRoot "sample_transcript.txt")
    foreach ($line in $lines) {
        $clean = $line -replace '^\s*\[[^\]]*\]\s*', ''
        if ($clean -match '^\s*([^:]+):\s*(.+)$') {
            $name = $matches[1].Trim()
            $text = $matches[2].Trim()
            if (-not $speakerVoice.ContainsKey($name)) {
                $idx = $i % $voices.Count
                $speakerVoice[$name] = $voices[$idx]
                $i++
            }
            try { $synth.SelectVoice($speakerVoice[$name]) } catch {}
            $synth.Speak($text)
        } elseif ($clean) {
            $synth.Speak($clean)
        }
    }
    $synth.SetOutputToNull()
    Write-Host "Wrote $Out"
} finally {
    $synth.Dispose()
}
