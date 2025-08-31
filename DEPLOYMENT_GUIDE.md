# 🚀 Deployment Guide - Real Estate Dashboard

**Developed by Maksim Kitikov - Upside Analytics**

## 📋 Prerequisites

- GitHub account: maksimkitikov
- SSH key generated: `~/.ssh/github_realestate_deploy`
- Render account (free tier)

## 🔑 Step 1: Add SSH Key to GitHub

### SSH Public Key:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIO5Hq75dYlP2b42CXWCJzrfRgslfg+KZwGBpE9TBSDZa maksimkitikov.uk@gmail.com
```

### Instructions:
1. Go to: https://github.com/maksimkitikov/realestate1/settings/keys/new
2. **Title:** `Real Estate Dashboard Deploy Key`
3. **Key:** Paste the SSH key above
4. **Allow write access:** ✅ Check this box
5. Click **"Add key"**

## 🐙 Step 2: Deploy to GitHub Repository

### Option A: Automatic Deployment
```bash
chmod +x deploy_to_github.sh
./deploy_to_github.sh
```

### Option B: Manual Deployment
```bash
# Initialize git (if not already done)
git init

# Add remote origin
git remote add origin https://github.com/maksimkitikov/realestate1.git

# Configure git user
git config user.name "Maksim Kitikov"
git config user.email "maksimkitikov.uk@gmail.com"

# Add and commit files
git add .
git commit -m "Deploy Real Estate Dashboard v2.0 - Maksim Kitikov - Upside Analytics"

# Push to GitHub
git push -u origin main
```

## ☁️ Step 3: Deploy to Render

### 3.1 Create Render Account
1. Go to https://render.com
2. Sign up with GitHub account
3. Authorize Render to access your repositories

### 3.2 Create Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect repository: `maksimkitikov/realestate1`
3. Configure settings:
   - **Name:** `realestate-dashboard`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python dashboard_advanced.py`
   - **Plan:** Free

### 3.3 Environment Variables
Add these environment variables in Render dashboard:

| Key | Value | Description |
|-----|-------|-------------|
| `DATABASE_URL` | `postgresql://user:password@host:port/database` | PostgreSQL connection string |
| `PYTHON_VERSION` | `3.11.0` | Python version |
| `FRED_API_KEY` | `your_fred_api_key` | FRED API key (optional) |

### 3.4 Database Setup
1. Create **PostgreSQL** database in Render
2. Copy the **Internal Database URL**
3. Set it as `DATABASE_URL` environment variable

## 🔧 Step 4: Configure Database

### 4.1 Database Schema
The application will automatically create tables using `schema.sql`

### 4.2 Data Population
Run the data pipeline:
```bash
python run_pipeline.py
```

## 🌐 Step 5: Access Your Application

Once deployed, your dashboard will be available at:
```
https://realestate-dashboard.onrender.com
```

## 📊 Features Available

- **Interactive US States Map**
- **Real-time Economic Data** (FRED API)
- **Regression Analysis** with R² scoring
- **Machine Learning Models**
- **Production-ready Platform**

## 🛠️ Troubleshooting

### Common Issues:

1. **Build Fails:**
   - Check Python version compatibility
   - Verify all dependencies in `requirements.txt`

2. **Database Connection Error:**
   - Verify `DATABASE_URL` environment variable
   - Check database credentials

3. **Application Won't Start:**
   - Check logs in Render dashboard
   - Verify start command: `python dashboard_advanced.py`

### Logs and Monitoring:
- View logs in Render dashboard
- Monitor application health at `/health` endpoint

## 📞 Support

**Developer:** Maksim Kitikov  
**Company:** Upside Analytics  
**Email:** maksimkitikov.uk@gmail.com

## 🔄 Continuous Deployment

The application is configured for automatic deployment:
- Any push to `main` branch triggers automatic deployment
- Render will rebuild and redeploy automatically

---

*Deployment Guide by Maksim Kitikov - Upside Analytics*
