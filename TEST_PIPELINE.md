# Testing Complete CI/CD Pipeline

## Changes Made for Testing

1. **app.py**: Added version variable `VERSION = "v1.0.1-test"` to track deployments
2. **k8s/deployment.yaml**: Changed image tag from `latest` to `master` for better tracking

## Step-by-Step Test Process

### Step 1: Commit and Push Changes

```bash
# Add all changes
git add app.py k8s/deployment.yaml

# Commit with descriptive message
git commit -m "Test CI/CD pipeline - update version and deployment tag"

# Push to trigger GitHub Actions
git push origin master
```

### Step 2: Monitor GitHub Actions (CI)

1. **Go to GitHub Repository**
   - URL: https://github.com/sherrygill2012-web/mspbackups_chatbot
   - Click on **"Actions"** tab

2. **Watch the Workflow Run**
   - You should see "Build and Push Docker Image" workflow running
   - Click on it to see detailed logs
   - Wait for it to complete (usually 3-5 minutes)

3. **Verify Build Success**
   - ✅ Green checkmark = Build successful
   - ❌ Red X = Build failed (check logs)

4. **Check Built Image**
   - Go to **Packages** tab in GitHub
   - Or visit: https://github.com/sherrygill2012-web/mspbackups_chatbot/pkgs/container/mspbackups_chatbot
   - You should see new image with tag: `master`

### Step 3: Monitor ArgoCD (CD)

1. **Access ArgoCD UI**
   - Get your ArgoCD URL (usually provided by your cluster admin)
   - Or port-forward: `kubectl port-forward svc/argocd-server -n argocd 8080:443`

2. **Check Application Status**
   - Login to ArgoCD UI
   - Find application: `msp360-chatbot`
   - Status should show:
     - 🟡 **Syncing** - ArgoCD detected the change
     - 🟢 **Healthy** - Deployment successful
     - 🔵 **Progressing** - New pods being created

3. **Watch Sync Process**
   - Click on the application
   - Click **"App Details"** tab
   - You'll see:
     - Git commit hash
     - Image being deployed
     - Pod status

4. **Verify Deployment**
   ```bash
   # Check pods
   kubectl get pods -n msp360
   
   # Check deployment
   kubectl get deployment msp360-chatbot -n msp360
   
   # Describe to see image
   kubectl describe deployment msp360-chatbot -n msp360 | grep Image
   ```

### Step 4: Verify the Change

1. **Check Running Pod**
   ```bash
   kubectl get pods -n msp360 -l app=msp360-chatbot
   ```

2. **Check Pod Logs** (optional)
   ```bash
   kubectl logs -n msp360 -l app=msp360-chatbot --tail=50
   ```

3. **Access Application** (if service is exposed)
   - Check service: `kubectl get svc -n msp360`
   - Access via port-forward or ingress

## Expected Timeline

```
Time    | Action
--------|----------------------------------
0:00    | git push origin master
0:01    | GitHub Actions workflow starts
2:00    | Docker image build completes
3:00    | Image pushed to GHCR
3:30    | ArgoCD detects Git change
4:00    | ArgoCD syncs deployment
5:00    | New pod starts with new image
6:00    | Pod becomes ready
```

## Troubleshooting

### GitHub Actions Not Running?
- ✅ Check Actions tab is enabled
- ✅ Verify workflow file is in `.github/workflows/`
- ✅ Check if you pushed to `master` branch
- ✅ Verify file paths match workflow triggers

### ArgoCD Not Syncing?
- ✅ Check ArgoCD application exists: `kubectl get application -n argocd`
- ✅ Verify Git repository URL is correct
- ✅ Check ArgoCD logs: `kubectl logs -n argocd -l app.kubernetes.io/name=argocd-application-controller`
- ✅ Manually sync: Click "Sync" button in ArgoCD UI

### Image Pull Errors?
- ✅ Verify image exists: Check GHCR packages
- ✅ Check imagePullSecrets: `kubectl get secret ghcr-secret -n msp360`
- ✅ Verify image tag matches: `master` tag should exist

### Pod Not Starting?
- ✅ Check pod events: `kubectl describe pod <pod-name> -n msp360`
- ✅ Check pod logs: `kubectl logs <pod-name> -n msp360`
- ✅ Verify secrets/configmap exist: `kubectl get secrets,configmap -n msp360`

## Success Indicators

✅ **GitHub Actions**: Green checkmark, image in Packages  
✅ **ArgoCD**: Application shows "Healthy" and "Synced"  
✅ **Kubernetes**: Pod running with new image tag  
✅ **Application**: App accessible and working  

## Next Steps After Testing

Once pipeline is verified:
1. Make real code changes
2. Push to trigger automatic build
3. ArgoCD will automatically deploy
4. Monitor both systems for issues

## Quick Commands Reference

```bash
# Check GitHub Actions status
gh run list  # If GitHub CLI installed

# Check ArgoCD applications
kubectl get applications -n argocd

# Check deployment status
kubectl rollout status deployment/msp360-chatbot -n msp360

# View ArgoCD sync status
argocd app get msp360-chatbot  # If ArgoCD CLI installed
```

