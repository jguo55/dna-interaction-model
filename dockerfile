FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN pip install --no-cache-dir numpy scikit-learn h5py

WORKDIR /workspace
COPY train.py /workspace/

CMD ["python", "train.py"]