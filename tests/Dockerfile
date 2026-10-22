FROM python:3.13-slim-bookworm

# everything the verifier needs is installed here, test.sh never installs anything
RUN pip install --no-cache-dir pytest==9.1.1 pytest-json-ctrf==0.5.2 numpy==2.2.6

# harbor does not upload tests/ at verify time, so bake it in
COPY . /tests/

# parent directory of the declared artifact
RUN mkdir -p /app
