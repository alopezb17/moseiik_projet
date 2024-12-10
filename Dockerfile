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

#Copy the files that we are going to use to our project 
COPY ./src /app/src
COPY Cargo.toml /app
COPY ./assets /app/assets
COPY ./tests/temp.rs /app/tests/temp.rs

#Download the datasets 
RUN wget https://nasext-vaader.insa-rennes.fr/ietr-vaader/moseiik_test_images.zip -P tests/
RUN mkdir tests/moseiik_test_images
RUN unzip tests/moseiik_test_images.zip -d tests/moseiik_test_images/


ENTRYPOINT [ "cargo", "test", "--release", "--" ]
