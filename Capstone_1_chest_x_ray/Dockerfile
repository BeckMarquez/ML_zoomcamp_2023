FROM python:3.9.18-slim

RUN pip install tflite-runtime==2.14.0
RUN pip install flask==3.0.0
RUN pip install waitress==2.1.2

WORKDIR /app

COPY ["predict.py", "chest_x_ray.tflite", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]