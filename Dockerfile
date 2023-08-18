FROM paidax/dev-containers:modelscope-v0.8

RUN pip install \
    fastapi \
    uvicorn \
    pydantic==1.10.8 \
    loguru && \
    rm -rf /root/.cache/pip/*

WORKDIR /home/fer

COPY ./download_model.py /home/fer/download_model.py

RUN python download_model.py

COPY . .

