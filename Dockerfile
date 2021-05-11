FROM python:3.8-slim as base
RUN apt-get update
ENV PYTHONFAULTHANDLER 1
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1


FROM base as deps
RUN apt-get install -y --no-install-recommends python3-dev \
  # OpenCV deps
  libglib2.0-0 libsm6 libxext6 libxrender1 \
  # Common deps
  g++

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -U pip

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install flask_cors


FROM base as final
RUN apt-get purge -y --auto-remove \
  && rm -rf /var/lib/apt/lists/*
# copy installed deps from dependencies image
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . /app
EXPOSE 8000
WORKDIR /app
CMD [ "/bin/bash", "-c", "gunicorn -c gunicorn_config.py application:app --bind 8000:8000" ]