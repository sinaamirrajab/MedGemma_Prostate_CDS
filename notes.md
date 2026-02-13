# How to run on VS Code Remote SSH

## 1. Install Node.js

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

## 2. Install deps and start Vite

```bash
cd /mnt/data9/projects/medgemma_challenge_UI/cds
npm install
npm run dev -- --host 0.0.0.0 --port 5173 --strictPort
```

## 3. Show app in VS Code (Remote SSH)

1. Open PORTS panel in VS Code.
2. Forward port 5173 (if not auto-forwarded).
3. Open the forwarded URL (usually http://127.0.0.1:5173) in your local browser.
4. You can also use Command Palette: Simple Browser: Show and open http://127.0.0.1:5173.