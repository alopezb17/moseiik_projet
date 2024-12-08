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

#Copy the files that we are going to use to our project 
COPY ./src /app/src
COPY Cargo.toml /app*



ENTRYPOINT [ "cargo", "test", "--release", "--" ]
