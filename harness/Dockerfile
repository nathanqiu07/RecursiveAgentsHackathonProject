FROM python:3.12-slim

# ResourceManager supplies a read-only root and a bounded writable /tmp tmpfs.
# Install experiment dependencies in a derived image before registering its pool.
COPY agent_harness/sandbox.py /opt/harness-sandbox.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp/workspace
WORKDIR /tmp
USER 65532:65532
ENTRYPOINT ["python", "-u", "/opt/harness-sandbox.py"]
