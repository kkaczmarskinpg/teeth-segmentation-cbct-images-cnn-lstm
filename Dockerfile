# Use Ubuntu as the base image
FROM docker.io/nvidia/cuda:12.0.0-base-ubuntu22.04

# Set environment variables to reduce TensorFlow warnings
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHON_VERSION=3.10

WORKDIR /ws

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    bash \
    git \
    wget \
    curl \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:\$PATH" >> /etc/profile.d/conda.sh

# Add Conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy the entire project into the working directory
COPY . /ws

RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya
    
# Install pip dependencies
RUN pip install --no-cache-dir -r requirements.txt && \ 
    pip install tensorflow[and-cuda]

# Expose ports (if using Jupyter or other services)
EXPOSE 8888 6006

# Default command to run your application (change as needed)
CMD ["/bin/bash"]
