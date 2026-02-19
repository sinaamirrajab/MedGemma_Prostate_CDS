# How to run on VS Code Remote SSH

## 1. Install Node.js (one-time setup)

Recommended without sudo: nvm

On remote server terminal:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"
nvm install --lts
node -v
npm -v
```

## 2. Install frontend dependencies (one-time setup)

```bash
cd /mnt/data9/projects/medgemma_challenge_UI/cds
npm install
```

---

## Every-time startup: run both servers

### Step A — Start the Python backends (MedGemma API + Classifier API)

**Backend 1 — MedGemma recommendation API (port 8000)**

Must be started with the `medgemma` conda environment:

```bash
nohup /mnt/data9/conda/medgemma/bin/python \
  /mnt/data9/projects/medgemma_challenge_UI/cds/backend/medgemma_api.py \
  > /tmp/medgemma_api.log 2>&1 &
echo "MedGemma PID: $!"
```

**Backend 2 — MedSigLIP classifier API (port 8001)**

Must be run from the `backend/` directory so model artifact paths resolve:

```bash
cd /mnt/data9/projects/medgemma_challenge_UI/cds/backend
nohup /mnt/data9/conda/medgemma/bin/python classifier_api.py \
  > /tmp/classifier_api.log 2>&1 &
echo "Classifier PID: $!"
```

Check both started correctly:

```bash
sleep 3 && tail -5 /tmp/medgemma_api.log
curl -s http://localhost:8001/health
```

You should see `Serving on 127.0.0.1:8000` and `{"status": "ok", ...}`.

> **To stop both backends:**
> ```bash
> pkill -f "medgemma_api.py"; pkill -f "classifier_api.py"
> ```

### Step B — Start the Vite frontend

Open a **separate terminal**, then:

```bash
cd /mnt/data9/projects/medgemma_challenge_UI/cds
npm run dev -- --host 0.0.0.0 --port 5173 --strictPort
```

> **Port 5173 already in use?** Run this first:
> ```bash
> fuser -k 5173/tcp
> ```

---

## 3. Open the app in your local browser (VS Code Remote SSH)

1. Open the **PORTS** panel in VS Code.
2. Forward port **5173** (usually auto-forwarded).
3. Open the forwarded URL — typically `http://127.0.0.1:5173` — in your local browser.
   - Alternatively: Command Palette → **Simple Browser: Show** → `http://127.0.0.1:5173`

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'transformers'` | Backend started with wrong Python. Kill it and restart using Step A with the full `/mnt/data9/conda/medgemma/bin/python` path. |
| `ECONNREFUSED 127.0.0.1:8000` | MedGemma API not running. Follow Step A (backend 1). |
| `Classifier failed: 500` or classifier button error | Classifier API not running, or started from wrong directory. Follow Step A (backend 2) — must `cd backend/` first. |
| `Port 5173 is already in use` | Run `fuser -k 5173/tcp` then retry Step B. |
| Images not loading | Ensure Vite is running and `vite.config.js` has `server.fs.strict: false`. |
