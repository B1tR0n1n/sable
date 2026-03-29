FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN pip install --no-cache-dir fastapi uvicorn websockets torch-geometric h5py

# Copy engine source (model definitions)
COPY engine/ /app/engine/

# Copy checkpoints, pre-computed scenarios, server
COPY checkpoints/ /app/checkpoints/
COPY scenarios/ /app/scenarios/
COPY server.py /app/
COPY sable_engine.py /app/
COPY dashboard.html /app/

EXPOSE 8080
CMD ["python", "server.py"]
