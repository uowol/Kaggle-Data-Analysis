# Use an official Ubuntu as a base image
FROM ubuntu:22.04
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Setup timezone and install dependencies
RUN echo 'Etc/UTC' > /etc/timezone \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && apt-get update \
    && apt-get -y --no-install-recommends install \
    tzdata build-essential curl libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget xz-utils coreutils \
    libxml2-dev libffi-dev liblzma-dev git ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv and Python
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
RUN curl https://pyenv.run | bash \
    && eval "$(pyenv init --path)" \
    && eval "$(pyenv init -)" \
    && pyenv install 3.12.8 && pyenv global 3.12.8

# Install poetry
ENV PATH="/root/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory and permissions
RUN mkdir -p /home/dev/data-analysis
WORKDIR /home/dev/data-analysis
COPY . /home/dev/data-analysis

# Install custom dependencies
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*