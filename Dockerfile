# 6TiSCH simulator development environment.
#
# The simulator is Python 2 code. Its pinned scientific stack (numpy 1.16.6,
# scipy 1.2.3, matplotlib 2.2.5, ...) only ships cp27 wheels for linux/x86_64,
# so we pin the platform rather than let pip try to build them from source.
FROM --platform=linux/amd64 python:2.7-slim-buster

# Debian archived the buster repositories; point apt at the archive so the
# image still builds.
RUN set -eux; \
    printf '%s\n' \
      'deb http://archive.debian.org/debian buster main' \
      'deb http://archive.debian.org/debian-security buster/updates main' \
      > /etc/apt/sources.list; \
    printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99no-check-valid-until; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc6-dev \
        make \
        git \
        ca-certificates; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /sim

# Install dependencies in their own layer so editing simulator code does not
# invalidate the (slow) dependency install.
COPY simulator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade "pip==20.3.4" "setuptools==44.1.1" "wheel==0.37.1" \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Matplotlib has no display inside the container; render to files instead.
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The repository is bind-mounted over this at run time (see docker-compose.yml),
# so the COPY here only matters if you build a standalone image.
COPY simulator /sim

CMD ["/bin/bash"]
