FROM ubuntu:24.04

# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md#run---mounttypecache
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt install -y --no-install-recommends nano ca-certificates python3 python3-pip python3-venv python3-flask wget curl gnupg2 gfortran libopenblas-dev liblapack-dev && \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
    apt update && \
    apt install -y google-chrome-stable && \
    apt -y upgrade && \
    apt autoremove -y

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# https://docs.docker.com/build/guide/mounts/#add-a-cache-mount
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked pip3 install opensearch-py gunicorn flask-compress humanize spacy sentence-transformers numpy flask-socketio selenium beautifulsoup4 duckduckgo-search langchain-community crewai[tools] pydantic

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s \
   CMD curl --fail localhost:5000/health || exit 1

COPY init-embedding.py /

RUN --mount=type=cache,target=/root/.cache/huggingface/hub/,sharing=locked python3 /init-embedding.py

COPY startup-dev.sh /
RUN chmod +x startup-dev.sh

COPY *.py /

WORKDIR /

ENV FLASK_APP=flaskapp

# add path to gunicorn
ENV PATH="${PATH}:/root/.local/bin/"

COPY flaskapp/ /flaskapp
