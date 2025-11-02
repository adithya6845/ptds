# Dockerfile for PTDS Streamlit app (includes system deps for osmnx/geopandas)
FROM python:3.10-slim

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin libgdal-dev \
    proj-bin libproj-dev \
    libspatialindex-dev \
    wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure GDAL env vars for pip builds
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Create app directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Upgrade pip and install wheels first
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Streamlit run (listen on all interfaces)
ENV STREAMLIT_SERVER_HEADLESS=true
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
