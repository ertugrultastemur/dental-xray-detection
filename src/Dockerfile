# Python version set
FROM python:3.10

# Working directory set
RUN mkdir -p /home/src
COPY . /home/src
WORKDIR /home/src

# Expose port
EXPOSE 8502


# Pip upgrade and install
RUN pip install --upgrade pip
RUN pip3 install --upgrade pip

# Timezone set
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install -r requirements.txt