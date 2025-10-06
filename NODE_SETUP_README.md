# Automatic Node.js PATH Setup

## 🎯 Problem Solved
This setup automatically checks for Node.js and adds it to PATH every time you open a terminal in VS Code.

---

## 📁 Files Added

### 1. `setup-node-path.ps1`
Automatically runs when you open a new PowerShell terminal in VS Code. It:
- ✅ Searches for Node.js in common installation locations
- ✅ Adds Node.js to PATH for the current session
- ✅ Shows Node.js version if found
- ❌ Shows installation instructions if not found

### 2. `.vscode/settings.json`
Configures VS Code to:
- Run `setup-node-path.ps1` automatically when opening PowerShell
- Add Node.js to PATH environment
- Set up Python and TypeScript configurations

---

## 🚀 How It Works

### When You Open VS Code:
1. Open any terminal (PowerShell)
2. The script automatically runs
3. If Node.js is installed, it's added to PATH ✅
4. You can immediately use `node`, `npm`, etc.

### Example Output (When Node.js is Found):
```
🔍 Checking Node.js setup...
✅ Found Node.js at: C:\Program Files\nodejs
✅ Added to PATH for this session
   Version: v20.x.x
🚀 Node.js is ready to use!
📦 npm version: 10.x.x
```

### Example Output (When Node.js is NOT Found):
```
🔍 Checking Node.js setup...
⚠️  Node.js not in current PATH, searching...
❌ Node.js is NOT installed on this system

📥 To install Node.js:
   1. Visit: https://nodejs.org/
   2. Download the LTS version (recommended)
   3. Run the installer (keep 'Add to PATH' checked)
   4. Restart VS Code after installation

   OR use winget:
   winget install OpenJS.NodeJS.LTS
```

---

## 📥 Installing Node.js (One-Time Setup)

### Option 1: Download from Official Website (Recommended)
1. Visit: **https://nodejs.org/**
2. Click the **green "LTS" button** (Long Term Support)
3. Run the downloaded installer (`.msi` file)
4. During installation:
   - ✅ Keep "Add to PATH" **CHECKED**
   - ✅ Use default installation location
5. Click through the installer
6. **Restart VS Code** after installation

### Option 2: Using Winget (Windows Package Manager)
```powershell
winget install OpenJS.NodeJS.LTS
```
Then restart VS Code.

### Option 3: Using Chocolatey
```powershell
choco install nodejs-lts -y
```
Then restart VS Code.

---

## ✅ Verify Installation

After installing Node.js, open a **NEW terminal** in VS Code:

```powershell
node --version
npm --version
```

You should see:
```
v20.x.x
10.x.x
```

---

## 🔧 Manual Setup (If Automatic Setup Doesn't Work)

If the automatic setup doesn't work, you can manually run the script:

```powershell
.\setup-node-path.ps1
```

Or add Node.js to PATH manually:
```powershell
$env:PATH = "C:\Program Files\nodejs;$env:PATH"
```

---

## 🏃 Running the Frontend

Once Node.js is set up:

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

The React app will open at: **http://localhost:3000**

---

## 🐛 Troubleshooting

### "node is not recognized" even after setup script runs
**Solution:** Node.js is not installed. Follow installation instructions above.

### Script doesn't run automatically
**Solution:** Run it manually:
```powershell
.\setup-node-path.ps1
```

### VS Code execution policy error
**Solution:** Allow script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Need to disable automatic script
**Solution:** Edit `.vscode/settings.json` and remove the `args` section from PowerShell profile.

---

## 📝 Notes

- **One-Time Installation:** Node.js only needs to be installed once
- **Automatic PATH Setup:** Script runs every time you open a terminal
- **No Admin Required:** Script only modifies session PATH, not system PATH
- **Works with:** Standard Node.js, nvm, or any Node.js installation

---

## 🎉 Benefits

✅ No more "node is not recognized" errors  
✅ Automatic setup on every terminal open  
✅ Works across different Node.js installation locations  
✅ Clear error messages with installation instructions  
✅ No manual PATH configuration needed  

---

**Status:** Ready to use! Just install Node.js once, and the script handles the rest automatically! 🚀
