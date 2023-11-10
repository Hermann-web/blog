---
date: 2023-11-10
authors: [hermann-web]
description: >
  This blog shows useful stuff to know about or learn to do as web developer or data scientist/engineer
  Always working the fastest and easiest ways toward success
categories:
  - software-and-tools
  - dev
  - OSX
  - linux
links:
  - setup/setting-up-a-blog.md
  - plugins/blog.md
title: "Run an application forever on linux made easy: Case of a java script"
---

## Introduction

If you're looking to turn your application into a background process, you have come to the right tutorial, always using the fastest way.

Instead of just writing theory, we we use a real world example i've worked on.

To run a Java application as a background process and keep it running forever, you can use a process manager like `systemd` on Linux. Here's how you can set up a `systemd` service to run your Java application:

Certainly! Here are the steps named as per their actions:


## Step 1: Create Service File
Create a new systemd service file for your Java application using a text editor:

```bash
sudo nano /etc/systemd/system/myapp.service
```

## Step 2: Configure Service
Paste the following configuration into the file, replacing `<jar-file-name>` with the name of your JAR file:

<!-- more -->


```ini
[Unit]
Description=My App

[Service]
User=myuser
ExecStart=/usr/bin/java -jar /path/to/myapp/<jar-file-name>.jar
SuccessExitStatus=143

[Install]
WantedBy=multi-user.target
```

## Step 3: Save and Exit
Save the file and exit the text editor.

## Step 4: Reload Daemon
Reload the systemd daemon to pick up the new service file:

```bash
sudo systemctl daemon-reload
```

## Step 5: Start Service
Start the new `myapp` service:

```bash
sudo systemctl start myapp
```

## Step 6: Check Service Status
Check the status of the service to make sure it's running:

```bash
sudo systemctl status myapp
```

If everything is set up correctly, you should see output indicating that the service is running. To stop the service, you can use the `sudo systemctl stop myapp` command.

With this setup, your Java application will run as a background process and automatically start when the server boots up. If the application crashes or stops for any reason, `systemd` will automatically restart it.
