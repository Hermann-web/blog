---
date: 2023-11-11
authors: [hermann-web]
comments: true
description: >
  This blog shows useful stuff to know about or learn to do as web developer or data scientist/engineer
  Always working the fastest and easiest ways toward success
categories:
  - dev
  - deployment
  - nginx
  - web
  - flask
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "Deploying any Web application with Nginx: Example of Flask"
---

## Introduction

You have created your flask application. How nice !
Now, you want to go a step further and deploy it.
For most hosting services, you have nice interfaces to deploy your python applications with support for flask.
But sometimes, you only have access via ssh to the server.

This is a very straigthforward tutorial on how to do it.

!!! note "This tutorial also applies to any web server you can run on local but want to deploy"

## Step 1: Install Nginx and Flask

Make sure you have Nginx and Flask installed on your server. If not, install them using the appropriate package manager for your operating system.

## Step 2: Configure Nginx

Create a new Nginx configuration file for your Flask app in the `/etc/nginx/sites-available/` directory. For example, you could name it `myapp.conf`. Edit the file and add the following configuration:

<!-- more -->

```conf
server {
    listen 80;
    server_name yourdomain.com;
    location / {
        proxy_pass http://localhost:5000; # assuming Flask is running on port 5000
        include /etc/nginx/proxy_params;
        proxy_redirect off;
    }
}
```

This tells Nginx to listen on port 80 (HTTP), forward all requests to your Flask app running on `localhost:5000`, and include some proxy parameters.

## Step 3: Create a symbolic link

Create a symbolic link from the `sites-available` directory to the `sites-enabled` directory by running the following command:

```bash
sudo ln -s /etc/nginx/sites-available/myapp.conf /etc/nginx/sites-enabled/
```

This will enable your Nginx configuration.

## Step 4: Test the configuration

Test your Nginx configuration by running the following command:

```bash
sudo nginx -t
```

If there are no errors, reload Nginx by running:

```bash
sudo service nginx reload
```

## Step 5: Start the Flask app

Start your Flask app by running the following command:

```bash
python app.py
```

Your Flask app should now be running and accessible through Nginx at `http://yourdomain.com`.

## exemple of flask app

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
```

## Related Posts

- [Mastering SSH and File Transfers to Remote servers: A Beginner's Handbook](../remote-access/ssh/access-remote-from-cli.md)
- [Navigating Redirect Challenges With GitHub Pages: A Creative Approach to Domain Migration](./github-action/github-action-redirection-app.md)
- [Run an application forever on linux made easy: Case of a javascript project](../OSX/linux/how-to-run-an-application-forever-on-linux.md)
