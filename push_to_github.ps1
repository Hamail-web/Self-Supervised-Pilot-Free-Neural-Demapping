# ============================================================
# push_to_github.ps1
# Run this script ONCE after installing Git for Windows.
# It initialises git, commits everything, and pushes to GitHub.
# ============================================================

# --- CONFIG (edit if needed) --------------------------------
$REPO_DIR   = "D:\litreture\Self-Supervised-Pilot-Free-Neural-Demapping"
$GIT_NAME   = "Hamail Ali"
$GIT_EMAIL  = "hamail1073@gmail.com"
$GITHUB_URL = "https://github.com/HamailAli-INRS/Self-Supervised-Pilot-Free-Neural-Demapping.git"
# ------------------------------------------------------------

Set-Location $REPO_DIR

git init
git config user.name  $GIT_NAME
git config user.email $GIT_EMAIL

# Exclude the large .pt checkpoint from git (uses Git LFS in real workflow)
# If you have Git LFS installed, uncomment the two lines below:
# git lfs install
# git lfs track "checkpoints/*.pt"

git add .
git commit -m "Initial release: self-supervised pilot-free neural demapping for 5G NR LDPC

Addresses all 5 reviewer points from Prof. Shervin Vakili (Nov 2026):
  1. Pseudo-label reliability: epsilon_thresh tightened to 0.12
  2. Pretraining + online adaptation narrative clarified
  3. 3-way ablation study added (Fixed / Self-sup. / Supervised)
  4. Dynamic channel tracking experiment added (AWGN->Rayleigh+Doppler)
  5. Exact Log-MAP AWGN baseline added; 2dB gain stated vs LS+LMMSE"

git branch -M main
git remote add origin $GITHUB_URL
git push -u origin main

Write-Host ""
Write-Host "SUCCESS: Repo pushed to $GITHUB_URL"
Write-Host ""
Write-Host "Next step: Add Prof. Vakili as collaborator:"
Write-Host "  GitHub.com -> Self-Supervised-Pilot-Free-Neural-Demapping"
Write-Host "  -> Settings -> Collaborators -> Add people"
Write-Host "  -> Enter: shervin.vakili@inrs.ca"
