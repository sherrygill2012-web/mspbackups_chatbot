# GitHub Actions Setup Guide

## Step-by-Step Instructions to Enable GitHub Actions

### Step 1: Verify Repository Connection ✅
Your repository is connected to: `https://github.com/sherrygill2012-web/mspbackups_chatbot.git`

### Step 2: GitHub Actions Workflow Created ✅
The workflow file has been created at: `.github/workflows/docker-build.yml`

This workflow will:
- Build Docker image when you push to `main` or `master` branch
- Build on pull requests
- Push images to GitHub Container Registry (GHCR)
- Tag images with branch name, commit SHA, and `latest`

### Step 3: Commit and Push the Workflow File

```bash
# Add the workflow file
git add .github/workflows/docker-build.yml

# Commit it
git commit -m "Add GitHub Actions workflow for Docker builds"

# Push to GitHub
git push origin master
```

### Step 4: Enable GitHub Actions (if not already enabled)

1. Go to your GitHub repository: `https://github.com/sherrygill2012-web/mspbackups_chatbot`
2. Click on the **"Actions"** tab
3. If you see a message about enabling Actions, click **"I understand my workflows, go ahead and enable them"**
4. GitHub Actions are enabled by default for public repos

### Step 5: Configure Package Permissions (if needed)

1. Go to your repository **Settings**
2. Click on **Actions** → **General**
3. Under **Workflow permissions**, ensure:
   - ✅ **Read and write permissions** is selected
   - ✅ **Allow GitHub Actions to create and approve pull requests** (optional)

### Step 6: Verify Package Registry Access

The workflow uses `GITHUB_TOKEN` which is automatically provided by GitHub Actions. It has permissions to:
- Read repository contents
- Write to GitHub Container Registry (GHCR)

### Step 7: Test the Workflow

1. Make a small change to trigger the workflow:
   ```bash
   # Edit any file (e.g., README.md)
   echo "# Test" >> README.md
   git add README.md
   git commit -m "Test GitHub Actions"
   git push origin master
   ```

2. Check the Actions tab:
   - Go to **Actions** tab in GitHub
   - You should see a workflow run starting
   - Click on it to see the build progress

### Step 8: View Built Images

After the workflow completes:

1. Go to your repository
2. Click on **Packages** (on the right sidebar)
3. Or visit: `https://github.com/sherrygill2012-web/mspbackups_chatbot/pkgs/container/mspbackups_chatbot`
4. You should see your Docker image with tags like:
   - `latest`
   - `master` (or `main`)
   - `master-<commit-sha>`

### Step 9: Update Deployment (Optional)

If you want to use specific tags instead of `latest`, update `k8s/deployment.yaml`:

```yaml
image: ghcr.io/sherrygill2012-web/mspbackups_chatbot:master
```

Or use commit SHA for more specific versions:
```yaml
image: ghcr.io/sherrygill2012-web/mspbackups_chatbot:master-abc1234
```

## Troubleshooting

### Workflow not running?
- Check if Actions are enabled in repository Settings → Actions
- Ensure you're pushing to `main` or `master` branch
- Check if the workflow file is in `.github/workflows/` directory

### Build fails?
- Check the Actions tab for error messages
- Verify Dockerfile is correct
- Check if all required files are present

### Permission errors?
- Go to Settings → Actions → General
- Ensure "Read and write permissions" is enabled
- Check if GITHUB_TOKEN has package write permissions

### Image not appearing in Packages?
- Wait a few minutes for the build to complete
- Check the Actions tab for any errors
- Verify the image name matches your repository name

## What Happens Next?

1. **Every push to main/master**: Docker image is built and pushed
2. **ArgoCD watches**: Your Git repository for changes
3. **When deployment.yaml changes**: ArgoCD syncs the new image to Kubernetes
4. **Result**: Automatic CI/CD pipeline! 🚀

## Complete CI/CD Flow

```
Code Push → GitHub Actions (Build) → GHCR (Store Image) → Git Commit → ArgoCD (Deploy) → Kubernetes
```

