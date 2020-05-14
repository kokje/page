FROM python:3.6.9-slim-stretch

# Each client has it's own dataset passed in at runtime
ARG datafile

# Install OpenJDK
# re: mkdir, https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=863199#23
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Create directory for source code and data
RUN mkdir -p /usr/src
RUN mkdir -p /usr/src/client
RUN mkdir -p /usr/data

# TODO: Replace with requirements.txt
# Install Hail
RUN pip3 --no-cache-dir install hail==0.2.37

# Install Flask
RUN pip3 --no-cache-dir install flask

#Install sklearn and pandas
RUN pip3 --no-cache-dir install sklearn
RUN pip3 --no-cache-dir install pandas

# Copy data silo to container
WORKDIR /usr/src/client
COPY $datafile /usr/data
COPY data/label_t2d.tsv /usr/data

# Schema for client db
COPY data/schema.sql /usr/data

# Client source code and config
COPY src/client/app.py .
COPY src/client/train.py .
COPY src/client/pca.py .
COPY src/client/db.py .
COPY src/client/client.cfg .

CMD ["python","app.py"]