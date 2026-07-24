# Hugging Face Space — a thin wrapper over the image published to ghcr.io.
#
# The base image already carries a trained model and runs the dashboard headless on
# 0.0.0.0, so there is nothing to build or download here: the Space just pulls it and
# routes to the port declared in this folder's README (app_port: 8501).
#
# Pinned to :latest so the Space tracks the newest release. Pin a version (e.g.
# :v0.2.1) for a frozen demo.
FROM ghcr.io/amayyas/athlete-injury-risk-detection:latest

# HF runs containers as uid 1000; the base image's `app` user is uid 1000, so this
# matches. Streamlit and matplotlib write only to /tmp (MPLCONFIGDIR is set upstream).
CMD ["dashboard", "--port", "8501"]
