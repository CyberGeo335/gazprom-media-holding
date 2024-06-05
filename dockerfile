FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

WORKDIR /code

COPY requirements.txt /code/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /code/

EXPOSE 5000

CMD [ "python", "app.py"]