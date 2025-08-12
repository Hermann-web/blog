---
date: 2023-11-27
authors: [hermann-web]
comments: true
title: Flask based File Hosting (web app & api & python module & cli app)
---

<!-- # File Sharing App -->
# Flask based File Hosting (web app & api & python module & cli app)

## Introduction

I've implemented A simple Flask application designed for sharing files.
The application can be hosted on a server so users can upload files and generate links for sharing.

## Prerequistes

- python >=3.9

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/hermann-web/simple-file-hosting-with-flask.git
   ```

2. Create a virtual environment and activate it:

   ```shell
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Start the Flask development server:

   ```
   python app.py
   ```

5. Access the application by visiting [http://localhost:5000](http://localhost:5000) in your web browser.

## Development Details

- The application allows users to upload files and share them with others by providing a link to the file.
- To use the application, users need to clone the [repository](https://github.com/Hermann-web/simple-file-hosting-with-flask), create a virtual environment, install the required dependencies, and start the Flask development server.
- Once the server is running, users can access the application by visiting <http://localhost:5000> in their web browser.
- The main page displays a list of shared files, and users can upload a file by clicking on "Upload a File" and selecting the file they want to share.
- The uploaded files will be listed on the main page for download.
- The application can be deployed on a remote server or cloud provider, such as AWS, Google Cloud, or Heroku.
- You can see the [step by step implementation of the code](../blog/posts/frameworks/web/flask/simple-file-hosting-with-flask-and-api-integration.md)

## Usage

### access the flask web

- The main page displays a list of shared files.
- To upload a file, click on "Upload a File" and select the file you want to share.
- The uploaded files will be listed on the main page for download.

### access the app through an api

- you can access the api with the routes `http://localhost:5000/api/*`
- The file [cli_app/cli_app.py](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/master/cli_app/cli_app.py) to access the api along with a context manager to handle sessions
- you can read the [api documentation](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/master/docs/api.md)

### access the app's api with a cli app

- The file [cli_app/sharefile.py](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/master/cli_app/sharefile.py) provide a cli app to access the api context manager
- Using your cli, you can get list, upload and download files. The api will be called behind the hood by [cli_app/cli_app.py](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/master/cli_app/cli_app.py)
- you can read the [api documentation](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/master/docs/cli-app.md)

## Deployment Guide

To deploy the File Sharing App, follow these steps:

1. Choose a remote server or cloud provider to host your application. Popular options include AWS, Google Cloud, and Heroku.

2. Set up an instance or virtual machine on your chosen server.

3. Connect to your remote server.

4. Install the required dependencies.

5. Modify the Flask application's configuration to use a production-ready web server.

6. Configure your domain or subdomain to point to the IP address of your remote server.

7. Set up SSL/TLS certificates for secure HTTPS communication.

8. Start the Flask application using the production-ready web server.

9. Verify that your file sharing app is accessible.

10. Monitor the deployed application for errors and performance issues.

Remember to follow best practices for securing your deployed application.

<!-- ## License

This project is licensed under the [MIT License](LICENSE). -->
