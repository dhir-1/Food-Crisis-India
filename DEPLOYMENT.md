# Deployment — Streamlit Community Cloud (Free)

Quick steps to deploy this repository (free) using Streamlit Community Cloud.

- Test locally:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

- Make sure your GitHub repo is **public** (Streamlit free tier requires public repos).
- Push your latest changes to `main` (already done).

- On Streamlit Cloud:
  1. Sign in at https://share.streamlit.io with your GitHub account.
 2. Click **New app** → choose your GitHub repo `Food-Crisis-India` → branch `main` → file path `app.py` → **Deploy**.

- App settings:
  - Add any runtime secrets via the app's **Settings → Secrets** (do not commit secrets into the repo).
  - If your app needs more RAM or private repos, consider paid plans — for public, small dashboards the free tier works fine.

- Large data note: avoid pushing large datasets (>100 MB) to GitHub. Use external storage (S3, Google Drive) and load at runtime.

If you want, I can also create a small GitHub Actions workflow or prepare a Hugging Face Space instead (also free for public apps). Tell me which and I'll add it.
