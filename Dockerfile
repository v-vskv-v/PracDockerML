FROM python:3.8-slim

COPY WebServerML/src/requirements.txt /root/WebServerML/src/requirements.txt

RUN chown -R root:root /root/WebServerML

WORKDIR /root/WebServerML/src
RUN pip3 install -r requirements.txt

COPY WebServerML/src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY mmf_317
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]
