```
FROM python:3.12-slim

# System tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    git \
    curl \
    ca-certificates \
    build-essential \
    libssl-dev \
    libffi-dev \
    r-base \
    less \
    bsdmainutils \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Core Python tooling you always want globally
RUN pip install --no-cache-dir \
    duckdb \
    pandas \
    pyarrow

# MinIO client (mc)
RUN curl -L https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc \
    && chmod +x /usr/local/bin/mc

# Default: drop you into a shell
CMD ["/bin/bash"]
```

