# 🚀 Deployment Instructions

All deployment files and documentation have been organized in the **`deployment/`** folder.

## 📁 What's Inside

```
deployment/
├── QUICKSTART.md          ← Start here for quick setup
├── README.md              ← Complete deployment guide
├── .env.example           ← Environment variables template
├── azure/                 ← Azure deployment scripts
├── docker/                ← Docker configurations
└── scripts/               ← Automation scripts
```

## ⚡ Quick Start

### For Team Members (Local Setup)
```powershell
.\deployment\scripts\setup-local.ps1
```

### For Docker Testing
```bash
cd deployment/docker
docker-compose up -d
```

### For Azure Production
```powershell
.\deployment\azure\deploy-app-service.ps1
```

## 📚 Full Documentation

**👉 See [deployment/QUICKSTART.md](./deployment/QUICKSTART.md) for quick start guide**

**👉 See [deployment/README.md](./deployment/README.md) for complete documentation**

---

**Questions?** Check the [deployment/README.md](./deployment/README.md) troubleshooting section.
