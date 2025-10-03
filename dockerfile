FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

RUN pip install --no-cache-dir numpy scikit-learn h5py

WORKDIR /workspace
COPY train.py /workspace/

CMD ["python", "train.py"]