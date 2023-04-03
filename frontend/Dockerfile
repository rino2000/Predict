FROM node:19.8.1-bullseye-slim

WORKDIR /frontend
COPY package.json .
RUN npm install
RUN npm i react-scripts && \
    npm i axios

COPY . ./
RUN npm run build
