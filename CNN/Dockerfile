# Use your working image that already has MinkowskiEngine 0.5.4 + PyTorch
FROM minkowski_engine

# Set sensible defaults for CPU threading; override at runtime if desired
ENV OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 \
    PYTHONUNBUFFERED=1

# Install minimal extras
RUN pip install --no-cache-dir tqdm
RUN pip install scikit-learn

# Workdir
WORKDIR /workspace

# Copy training script
COPY train.py /workspace/train.py

# Default command: print help
CMD ["python3", "train.py", "--help"]
