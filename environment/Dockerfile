FROM python:3.13-slim-bookworm

RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 scikit-learn==1.6.1

ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

WORKDIR /app
COPY data/ /app/
