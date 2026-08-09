FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2

ARG OCI_CREATED
ARG OCI_REVISION

LABEL org.opencontainers.image.source="https://github.com/seg-opt/seg-opt" \
      org.opencontainers.image.revision="$OCI_REVISION" \
      org.opencontainers.image.created="$OCI_CREATED" \
      org.opencontainers.image.description="PSNC Eagle container runtime diagnostic image"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY containers/requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

WORKDIR /opt/seg-opt
COPY --chown=1000:1000 containers/smoke_test.py /opt/seg-opt/smoke_test.py

USER 1000:1000
CMD ["python3", "/opt/seg-opt/smoke_test.py"]