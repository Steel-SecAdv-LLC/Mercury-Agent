FROM python:3.12-slim

# Immediate security patch — updates all system packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (extra safety)
ARG USERNAME=omniava
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Switch to non-root + clean pip installs
USER $USERNAME
WORKDIR /app

COPY --chown=$USERNAME:$USER_GID requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=$USERNAME:$USER_GID . .

CMD ["python", "src/mercury/train.py"]
