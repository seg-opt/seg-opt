# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.3@sha256:dfd1e6972e100ca2fbf1f391effc3dd4aa57f319bf03c3e321e0a3f3341ed5af AS uv

FROM nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:605fb0c8acf8674e164d822da8a8521f3a655056e569f0899e72ae940e1fe7dc

ARG OCI_CREATED
ARG OCI_REVISION

LABEL org.opencontainers.image.source="https://github.com/seg-opt/seg-opt" \
      org.opencontainers.image.revision="$OCI_REVISION" \
      org.opencontainers.image.created="$OCI_CREATED" \
      org.opencontainers.image.description="DINOv3 development environment for PSNC Eagle"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/seg-opt-env/.venv \
    PATH="/opt/seg-opt-env/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    WANDB_CACHE_DIR=/cache/wandb/cache \
    WANDB_CONFIG_DIR=/cache/wandb/config \
    WANDB_DATA_DIR=/cache/wandb/data

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /cache/tmp /output /workspace \
    && chown -R 1000:1000 /cache /output /workspace

ENV TMPDIR=/cache/tmp

COPY --from=uv /uv /uvx /bin/

WORKDIR /opt/seg-opt-env
COPY pyproject.toml uv.lock .python-version ./
RUN uv python install 3.13 \
    && uv sync --frozen --no-install-project --all-groups \
    && uv cache clean

COPY --chown=1000:1000 containers/development_smoke_test.py /opt/seg-opt-env/development_smoke_test.py

WORKDIR /workspace
USER 1000:1000
CMD ["python", "/opt/seg-opt-env/development_smoke_test.py"]