
FROM gcr.io/deeplearning-platform-release/tf-cpu.2-13
RUN pip install -U pandas==2.1.4 numpy==1.24.3 fire cloudml-hypertune scikit-learn==1.3.2 fsspec gcsfs
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
