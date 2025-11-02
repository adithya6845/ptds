PTDS app — Deployment notes

This repository contains a Streamlit app that uses OSMnx/NetworkX/Folium for route visualization.

Two recommended deployment options:

1) Docker (recommended — reproducible and handles system deps)
--------------------------------------------------------------
Build locally and run:

# from project root (PowerShell)
docker build -t ptds-app:latest .
docker run -p 8501:8501 ptds-app:latest

Then open http://localhost:8501

To push to a container registry and deploy on a cloud provider, tag and push the image and use your provider's container service.

2) Streamlit Cloud (quick, but may fail for heavy geospatial libs)
-----------------------------------------------------------------
Streamlit Cloud installs from `requirements.txt`. However, `osmnx` and `geopandas` often need system libraries (GDAL/PROJ) that the hosted environment may not provide. If you want to try Streamlit Cloud, ensure `requirements.txt` is committed and:

- Go to Streamlit Cloud, create a new app from this repo/branch.
- If pip install fails for geopandas/osmnx, prefer Docker deployment.

Troubleshooting
---------------
- If you see ModuleNotFoundError for `networkx` or other packages: ensure `requirements.txt` contains the package and the deployment built successfully.
- If you see build errors installing `geopandas`/`osmnx`: these require system libraries; use Docker to avoid build-time issues.

If you'd like I can:
- Try a local build here and capture output logs.
- Create a deployment script for a specific provider (Heroku/DigitalOcean/Azure).

Tell me which option you want me to run next: `build-local` (I will run the docker build here), `prepare-streamlit` (I will further slim requirements for Streamlit Cloud), or `create-heroku` (I will generate a Procfile and requirements tuned for Heroku).