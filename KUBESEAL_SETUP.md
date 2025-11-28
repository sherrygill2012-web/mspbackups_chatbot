# Kubeseal Secret Management Guide

This guide explains how to manage Kubernetes secrets securely using **Sealed Secrets (Kubeseal)** for GitOps workflows with ArgoCD.

## Overview

Kubeseal encrypts Kubernetes Secrets into **SealedSecrets**, which are safe to store in Git. The Sealed Secrets controller running in your cluster decrypts them back into regular Secrets.

```
┌─────────────────┐     kubeseal      ┌──────────────────┐
│  Plain Secret   │ ───────────────▶  │  SealedSecret    │ ──▶ Git
│  (local only)   │    (encrypt)      │  (encrypted)     │
└─────────────────┘                   └──────────────────┘
                                              │
                                              ▼
                                      ┌──────────────────┐
                                      │ Sealed Secrets   │
                                      │ Controller       │
                                      │ (in cluster)     │
                                      └──────────────────┘
                                              │
                                              ▼ (decrypt)
                                      ┌──────────────────┐
                                      │  Regular Secret  │
                                      │  (in cluster)    │
                                      └──────────────────┘
```

---

## Prerequisites

- Kubernetes cluster (K3s, K8s, etc.)
- `kubectl` configured to access your cluster
- ArgoCD installed (optional, for GitOps)

---

## Step 1: Install Sealed Secrets Controller

Install the controller in your cluster:

```bash
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.5/controller.yaml
```

Verify it's running:

```bash
kubectl get pods -n kube-system -l name=sealed-secrets-controller
```

Expected output:
```
NAME                                         READY   STATUS    RESTARTS   AGE
sealed-secrets-controller-xxxxxxxxx-xxxxx    1/1     Running   0          1m
```

---

## Step 2: Install Kubeseal CLI

### macOS (Homebrew)
```bash
brew install kubeseal
```

### Linux
```bash
# Download binary
KUBESEAL_VERSION=0.24.5
curl -LO "https://github.com/bitnami-labs/sealed-secrets/releases/download/v${KUBESEAL_VERSION}/kubeseal-${KUBESEAL_VERSION}-linux-amd64.tar.gz"

# Extract and install
tar -xzf kubeseal-${KUBESEAL_VERSION}-linux-amd64.tar.gz
sudo mv kubeseal /usr/local/bin/
rm kubeseal-${KUBESEAL_VERSION}-linux-amd64.tar.gz
```

### Verify installation
```bash
kubeseal --version
```

---

## Step 3: Create a Secret Template

Create a regular Kubernetes Secret file (this stays local, never commit it):

```yaml
# /tmp/my-secret.yaml (DO NOT COMMIT THIS FILE)
apiVersion: v1
kind: Secret
metadata:
  name: msp360-chatbot-secrets
  namespace: msp360
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-your-actual-api-key"
  GEMINI_API_KEY: "your-actual-gemini-key"
```

---

## Step 4: Seal the Secret

Convert the plain secret to a SealedSecret:

```bash
kubeseal --format yaml < /tmp/my-secret.yaml > k8s/sealed-secret.yaml
```

The output file `sealed-secret.yaml` is safe to commit to Git.

**Important:** Delete the plain secret immediately:
```bash
rm /tmp/my-secret.yaml
```

---

## Step 5: Apply or Commit the SealedSecret

### Option A: Direct Apply
```bash
kubectl apply -f k8s/sealed-secret.yaml
```

### Option B: GitOps with ArgoCD
```bash
git add k8s/sealed-secret.yaml
git commit -m "Add sealed secret for API keys"
git push
```

ArgoCD will automatically sync and create the secret.

---

## Step 6: Verify the Secret

Check that both the SealedSecret and decrypted Secret exist:

```bash
kubectl get sealedsecret,secret -n msp360 | grep msp360-chatbot
```

Expected output:
```
sealedsecret.bitnami.com/msp360-chatbot-secrets   1m
secret/msp360-chatbot-secrets                     1m
```

View the secret data (base64 encoded):
```bash
kubectl get secret msp360-chatbot-secrets -n msp360 -o jsonpath='{.data}'
```

---

## Updating Secrets

To update an existing sealed secret:

```bash
# 1. Create new plain secret
cat > /tmp/updated-secret.yaml << 'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: msp360-chatbot-secrets
  namespace: msp360
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-new-api-key"
  GEMINI_API_KEY: "new-gemini-key"
EOF

# 2. Seal it (overwrites existing)
kubeseal --format yaml < /tmp/updated-secret.yaml > k8s/sealed-secret.yaml

# 3. Delete plain secret
rm /tmp/updated-secret.yaml

# 4. Commit and push
git add k8s/sealed-secret.yaml
git commit -m "Update API keys"
git push
```

---

## Adding New Keys

To add a new key to an existing secret, include ALL keys in the template:

```yaml
# /tmp/secret-with-new-key.yaml
apiVersion: v1
kind: Secret
metadata:
  name: msp360-chatbot-secrets
  namespace: msp360
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-existing-key"
  GEMINI_API_KEY: "existing-gemini-key"
  NEW_API_KEY: "new-key-value"        # Added
```

Then seal and commit as usual.

---

## Backup & Recovery

### Export the Sealing Key (for backup)

```bash
kubectl get secret -n kube-system -l sealedsecrets.bitnami.com/sealed-secrets-key \
  -o yaml > sealed-secrets-master-key.yaml
```

**Store this file securely!** If you lose it, you cannot decrypt existing SealedSecrets.

### Restore from Backup

```bash
kubectl apply -f sealed-secrets-master-key.yaml
kubectl delete pod -n kube-system -l name=sealed-secrets-controller
```

---

## Fetching the Public Key

If you need to seal secrets offline (without cluster access):

```bash
kubeseal --fetch-cert > sealed-secrets-pub.pem
```

Then seal using the certificate:
```bash
kubeseal --cert sealed-secrets-pub.pem --format yaml < secret.yaml > sealed-secret.yaml
```

---

## Troubleshooting

### SealedSecret not creating Secret

Check controller logs:
```bash
kubectl logs -n kube-system -l name=sealed-secrets-controller
```

### "no key could decrypt" error

The sealing key has been rotated or the controller was reinstalled. You need to re-seal the secret:
```bash
# Re-seal with current cluster key
kubeseal --format yaml < /tmp/plain-secret.yaml > k8s/sealed-secret.yaml
```

### Namespace mismatch

SealedSecrets are bound to a specific namespace by default. If you see decryption errors, ensure the namespace in metadata matches where you're applying it.

---

## Security Best Practices

1. **Never commit plain secrets** - Only commit `sealed-secret.yaml` files
2. **Backup your sealing key** - Without it, you can't decrypt existing SealedSecrets
3. **Use separate secrets per environment** - Different clusters have different sealing keys
4. **Rotate secrets regularly** - Re-seal and commit new values
5. **Add to .gitignore**:
   ```gitignore
   # Plain secrets - NEVER commit
   *secret*.local.yaml
   /tmp/*.yaml
   
   # Allow sealed secrets
   !k8s/sealed-secret.yaml
   ```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `kubeseal --format yaml < secret.yaml > sealed.yaml` | Seal a secret |
| `kubeseal --fetch-cert > pub.pem` | Get public key for offline sealing |
| `kubectl get sealedsecret -A` | List all SealedSecrets |
| `kubectl logs -n kube-system -l name=sealed-secrets-controller` | View controller logs |

---

## Files in This Project

| File | Purpose |
|------|---------|
| `k8s/sealed-secret.yaml` | Encrypted SealedSecret (safe to commit) |
| `k8s/kustomization.yaml` | References sealed-secret.yaml |
| `k8s/argocd-application.yaml` | ArgoCD app config (no special plugins needed) |

