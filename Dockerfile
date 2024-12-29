#Upload rust 
FROM rust:latest
LABEL authors="PAULINE et ANAMARIA"

#Create the workfile directory of the container
WORKDIR /app

# Update package list and install software
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install wget -y
RUN apt-get install unzip -y
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the RUST_BACKTRACE environment variable
ENV RUST_BACKTRACE=1

#Copy source code and configuration files
COPY ./src /app/src
COPY Cargo.toml /app
COPY ./tests/temp.rs /app/tests/temp.rs
COPY ./assets /app/assets

# Define the dataset path and url
ENV DATASET_DIR=assets/moseiik_test_images
ENV DATASET_URL=https://nasext-vaader.insa-rennes.fr/ietr-vaader/moseiik_test_images.zip

#Download the datasets if they are not already downloaded
#It enables executing the tests in local and remote environments
RUN if [ ! -d ${DATASET_DIR} ]; then \
    mkdir ${DATASET_DIR} && \
    wget ${DATASET_URL} -P assets/ && \
    unzip assets/moseiik_test_images.zip -d ${DATASET_DIR}; \
  fi

# Set the entry point to run the tests
ENTRYPOINT [ "cargo", "test", "--release", "--" ]
