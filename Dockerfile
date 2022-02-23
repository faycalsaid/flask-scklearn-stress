FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./data ./data
COPY ./main.py ./main.py

CMD [ "python3", "main.py" , "8080"]