$ErrorActionPreference = "Stop"
$questions = @(
  "Tuyên ngôn Độc lập được đọc ở đâu?",
  "Chiến dịch Điện Biên Phủ kết thúc ngày nào?",
  "Hiệp định Paris 1973 có ý nghĩa gì?",
  "hôm nay trời mưa không?"
)
foreach ($q in $questions) {
  $body = @{
    message = $q
    data_profile = "cloud_primary_final"
    force_cloud_llm_final = $true
    session_id = "github_release_smoke"
  } | ConvertTo-Json
  $res = Invoke-RestMethod -Uri "http://127.0.0.1:31114/9router-fast-chat" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 90
  Write-Host "`nQ: $q"
  Write-Host "A: $($res.answer)"
}

