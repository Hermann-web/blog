---
date: 2024-03-21
authors: [hermann-web]
comments: true
description: >
  Unlock the full potential of Docker with this comprehensive guide, covering essential commands, best practices, and advanced techniques for efficient container management.
categories:
  - OSX
  - docker
  - containers
  - deployment
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "Mastering Docker: A Comprehensive Guide to Efficient Container Management"
---

# Mastering Docker: A Comprehensive Guide to Efficient Container Management

__Tired of complicated installation processes, inconsistent environments, and tedious deployment procedures in your development workflow?__

<div class="float-img-container float-img-right">
  <a title="Credit: docker.com" href="https://www.docker.com/company/newsroom/media-resources/"><img style="width:150px" alt="docker-logo" src="https://driftt.imgix.net/https%3A%2F%2Fdriftt.imgix.net%2Fhttps%253A%252F%252Fs3.us-east-1.amazonaws.com%252Fcustomer-api-avatars-prod%252F5244849%252F2992b5064fb40c8454b29d7dde843e02wxmbzt425w27%3Ffit%3Dmax%26fm%3Dpng%26h%3D200%26w%3D200%26s%3Dafd7b37a323ef1171abe7d37c7b901e5?fit=max&fm=png&h=200&w=200&s=f2525fe124159f60cc588983058867b1g"></a>
</div>

Whether you're an experienced developer or just starting out, Docker offers a game-changing approach when it comes to application deployment, scalability, and consistency.

Dive into the world of Docker and utilise its full potential to simplify your container management and increase your productivity!

In this comprehensive guide, we will introduce you to Docker's essential commands, best practices, and advanced techniques to help you harness the power of containerisation effectively.

<!-- more -->

Ready to embark on a journey towards transparent development, easy deployment, and unrivalled efficiency? Let's dive in and master Docker from A to Z!

## Understanding Docker: A Primer on Containerization

Before diving into Docker's intricacies, let's establish a foundational understanding of containerization:

### What is Docker?

Docker is a platform designed to make it easier to create, deploy, and run applications by using containers. Containers allow developers to package an application with all of its dependencies into a standardized unit for software development.

### Key Concepts

- __Images:__ Lightweight, standalone, and executable packages that contain everything needed to run a piece of software, including the code, runtime, libraries, and dependencies.
- __Containers:__ Runnable instances of Docker images, encapsulating the software and its dependencies, ensuring consistency across different environments.
- __Dockerfile:__ A text document that contains all the commands a user could call on the command line to assemble an image.
- __Docker Hub:__ A cloud-based repository provided by Docker for finding, sharing, and distributing container images.

With this foundation in place, let's delve deeper into Docker's functionalities and explore .

## Essential Docker Commands: Navigating the Docker Universe

Creating a container from an image or using a specific distribution like Ubuntu involves a few steps. Here are the basic commands to create a Docker container:

### Create a Container from an Image

#### Pull an Image from Docker Hub

If you haven't already downloaded the image you want to use, you can pull it from Docker Hub using `docker pull`:

```bash
docker pull <image_name>
```

Replace `<image_name>` with the name of the image you want to pull. For example, to pull the Ubuntu image:

```bash
docker pull ubuntu
```

#### Run a Container from an Image

Once you have the image, you can create a container by running it with `docker run`:

```bash
docker run -it --name <container_name> <image_name> /bin/bash
```

Explanation of the flags used:

- `-it`: Starts the container in interactive mode with a terminal.
- `--name <container_name>`: Assigns a specific name to your container.
- `<image_name>`: Specifies the image to use for creating the container.
- `/bin/bash` (or another command): Starts a specific command or shell in the container. For Ubuntu, using `/bin/bash` opens a Bash shell.

For instance, to create a container named `my-ubuntu-container` from the Ubuntu image:

```bash
docker run -it --name my-ubuntu-container ubuntu /bin/bash
```

This will start a new container based on the Ubuntu image and give you access to the Bash shell within that container.

### Install Software or Customize the Container

Once inside the container, you can install software, make changes, or configure it as needed using standard commands as you would on a regular Linux system. For example, within the container:

```bash
apt update
apt install <package_name>
```

Replace `<package_name>` with the name of the package you want to install.

### Commit Changes to a New Image (Optional)

If you make changes within the container and want to save them as a new image, you can commit the container's current state:

```bash
docker commit <container_id> <new_image_name>
```

Replace `<container_id>` with the ID of your container, and `<new_image_name>` with the name you want to give to the new image.

For example:

```bash
docker commit my-ubuntu-container my-custom-ubuntu
```

This will create a new image named `my-custom-ubuntu` based on the changes made in the `my-ubuntu-container` container.

Remember, changes made inside a container will be lost if you remove the container without committing those changes to a new image.

These steps should help you create, customize, and manage containers based on Docker images, allowing you to work with specific distributions like Ubuntu or any other image available on Docker Hub.

### Get all container along with their ID and status

To access an existing container from another terminal, you need its container ID or name. You can find this information looking for all containers.

To get the container ID, you can use the following command in the terminal:

```bash
$ docker ps
# to see all containers, running and stopped
$ docker ps -a
```

This command lists all the running containers along with their IDs, names, and other details.

Example output ([nodejs-docker](https://nodejs.org/en/docs/guides/nodejs-docker-webapp)):

```plaintext
ID            IMAGE                                COMMAND    ...   PORTS
ecce33b30ebf  <your username>/node-web-app:latest  npm start  ...   49160->8080
```

### Print app output

To print the application output, you can use the following command:

```bash
docker logs <container id>
```

Example output:

```plaintext
Running on http://localhost:8080
```

### Access the application

To access the application, you can use the `curl` command:

```bash
curl -i localhost:49160
```

Example output:

```plaintext
HTTP/1.1 200 OK
X-Powered-By: Express
Content-Type: text/html; charset=utf-8
Content-Length: 12
ETag: W/"c-M6tWOb/Y57lesdjQuHeB1P/qTV0"
Date: Mon, 13 Nov 2017 20:53:59 GMT
Connection: keep-alive

Hello world
```

### Access cmd line of the running container

Once you have the container ID or name, you can use the `docker exec` command to access it from another terminal. The command syntax is:

```bash
docker exec -it <container_id_or_name_> /bin/bash
```

Replace `<container_id_or_name>` with the actual ID or name of your container.

For example, if your container's name is `my-ubuntu-container`, you'd run:

```bash
docker exec -it my-ubuntu-container /bin/bash
```

This command will open a new terminal session inside the running container, allowing you to interact with it just like you did from the initial terminal.

Remember, you can access a running container from multiple terminals simultaneously using `docker exec`. Each terminal session will have its own instance of a shell within the same container, enabling parallel interactions and commands execution.

### Kill the running container

To stop the running container, you can use the following command:

```bash
docker kill <container id>
```

Example output:

```plaintext
<container id>
```

### Restart a stopped/killed container

To run the killed container, you can use the following command:

```bash
# see all the containers
$ docker ps -a 
# restart the one
$ docker start <container id>
```

Example output:

```plaintext
<container id>
```

### Confirm that the app has stopped

To confirm that the application has stopped, you can use the `curl` command again:

```bash
curl -i localhost:49160
```

Example output:

```plaintext
curl: (7) Failed to connect to localhost port 49160: Connection refused
```

### delete a container

To delete a Docker container by its ID, you can use the `docker rm` command followed by the container ID. Here's the syntax:

```bash
docker rm <container_id>
```

Replace `<container_id>` with the actual ID of the container you want to delete.

For example, if your container ID is `a460131e5352`, you'd run:

```bash
docker rm a460131e5352
```

This command will remove the specified container. Make sure the container is stopped before attempting to remove it. If the container is running, you can stop it using `docker stop <container_id>` before removing it.

Please note that deleting a container is a permanent action, and its associated data (unless stored in a separate volume) will be lost.

## Share files between your PC and a Docker container

To share files between your PC and a Docker container using the container ID, you can use Docker's `docker cp` command. This command allows you to copy files or directories between the host system and a container. Here's how you can use it:

### Copying Files from Host to Container

```bash
docker cp /path/on/host/file_or_directory <container_id>:/path/in/container
```

Replace `/path/on/host/file_or_directory` with the path to the file or directory on your host system that you want to copy into the container.

Replace `<container_id>` with the ID of the container where you want to copy the files.

Replace `/path/in/container` with the path inside the container where you want to place the files.

For instance, if you have a file `/home/user/file.txt` on your host system and you want to copy it into a container with ID `a460131e5352` into the container's `/app/data` directory, you'd use:

```bash
docker cp /home/user/file.txt a460131e5352:/app/data
```

### Copying Files from Container to Host

Similarly, you can copy files from a container back to your host system:

```bash
docker cp <container_id>:/path/in/container/file_or_directory /path/on/host
```

Replace `<container_id>` with the ID of the container from which you want to copy files.

Replace `/path/in/container/file_or_directory` with the path inside the container of the file or directory you want to copy.

Replace `/path/on/host` with the path on your host system where you want to place the copied files.

For example, if you want to copy a file named `output.log` from a container with ID `a460131e5352` located at `/var/logs` to your host's `/home/user/logs` directory, you'd run:

```bash
docker cp a460131e5352:/var/logs/output.log /home/user/logs
```

This way, you can share files between your PC and a Docker container using the `docker cp` command.

## VSCode Tools

!!! note "open a container folder in vscode"
    consult [this link](https://code.visualstudio.com/docs/devcontainers/containers)

## Create an Image from an Application

Managing Docker images and their containers involves a structured process. Here's an organized version detailing the steps:

### **Organize Files:**

Organizing files is pivotal for smooth Docker image creation. Ensure your application code, including the Dockerfile, `.dockerignore`, and necessary configuration files, resides within the same directory. For instance, a typical Node.js app might have the following structure:

```plaintext
your-app-directory/
├── Dockerfile
├── .dockerignore
├── package.json
├── index.js (or main application file)
├── ... (other application files and directories)
```

Keeping all pertinent files together simplifies referencing in the Dockerfile and ensures that only essential application files are included in the image.

### **Example Dockerfile for an Express App:**

Here's a simple Dockerfile suitable for an Express application running on Node.js 14:

```Dockerfile
# Use Node.js 14 as the base image
FROM node:14

# Create and set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install application dependencies
RUN npm install

# Copy the entire application directory into the container
COPY . .

# Expose port 3001 to the outside world
EXPOSE 3001

# Define the command to start the application
CMD ["npm", "start"]
```

### **Create .dockerignore:**

A sample `.dockerignore` might contain exclusions like:

```plaintext
node_modules
npm-debug.log
images
doc-api
logs
.github
.git
.env
.prod.env
tests
```

The `.dockerignore` file excludes unnecessary files and directories from being copied into the Docker image during the build process. Including `.git` is generally advised to prevent large, unneeded directories from being included.

Additionally, consider excluding:

- __Sensitive Data:__ Files containing sensitive information.
- __Development Configs:__ Configuration specific to local development.
- __Build Artifacts:__ Generated files not needed for the running application.
- __IDE/Editor Files:__ Editor-specific files not crucial for the app's runtime.
- __Documentation/Images:__ Non-essential assets irrelevant to the app's functionality.

#### Quick trick: Using a modified .gitignore

Consider using a modified `.gitignore` as `.dockerignore`. However, ensure it's not overly restrictive, as some files ignored in version control might be necessary for the app to run correctly within a Docker container.

In my example, `.test.env` was in .gitignore but important. However, `tests` is not in .gitignore but useless in production

### **Build the Image:**

Navigate to the directory containing your Dockerfile and execute:

```bash
docker build -t your-image-name .
```

This command constructs a Docker image from the Dockerfile and tags it with the specified name.

### **Run a Container from the New Image:**

Start a container using the newly built image:

```bash
docker run -p 49160:3001 your-image-name
```

Replace `your-image-name` with the image's name. If your app runs on port 3001 in the container, access it at `localhost:49160` on your machine.

## Managing Images and Associated Containers

### List All Images

To view all Docker images on your system, execute:

```bash
docker images
```

Use `-q` for a compact list displaying only image IDs:

```bash
docker images -q
```

### List Containers Using a Specific Image

To list containers using a particular image:

```bash
docker ps -a --filter ancestor=your-image-name
```

Use `--format` to customize output (e.g., `{{.ID}}` for IDs):

```bash
docker ps -a --filter ancestor=your-image-name --format "{{.ID}}"
```

### Stop Containers Using an Image

Stop all containers associated with an image:

```bash
docker stop $(docker ps -a -q --filter ancestor=your-image-name)
```

### Remove an Image and Associated Containers

To rebuild an image, follow these steps:

1. __Stop and Remove Containers__:

   ```bash
   docker stop $(docker ps -a -q --filter ancestor=your-image-name)
   docker rm $(docker ps -a -q --filter ancestor=your-image-name)
   ```

   Replace `your-image-name` with the specific image name or ID.

2. __Remove the Image__:

   ```bash
   docker rmi your-image-name
   ```

   Replace `your-image-name` with the name of your Docker image.

3. __Rebuild the Image__:

   After removing the container and image, rebuild the image:

   ```bash
   docker build -t your-image-name .
   ```

   This rebuilds the Docker image using the Dockerfile in the directory and tags it with the specified name.

4. __Run a New Container__:

   Start a new container from the rebuilt image:

   ```bash
   docker run -p 49160:3001 your-image-name
   ```

   Replace `your-image-name` with the name of your newly built Docker image.

By following these steps, you'll manage existing containers, remove associated images, and rebuild a fresh image for running a new container with updated changes.

### clean env

#### remove all images that have none as img name

```bash
#!/bin/bash

# Get a list of image IDs for images with the tag "none"
images=$(docker images | grep "none" | awk '{print $3}')

# Loop through each image ID and execute the commands
for image_id in $images
do
 # Stop containers based on the image
 docker stop $(docker ps -a -q --filter ancestor=$image_id)
 
 # Remove containers based on the image
 docker rm $(docker ps -a -q --filter ancestor=$image_id)
 
 # Remove the image itself
 docker rmi $image_id
done
```

#### remove all images that dont have a container

```bash
# Get a list of image IDs for all images
images=$(docker images | awk '{print $3}')

# Loop through each image ID and execute the commands
for image_id in $images
do
 # Remove the image itself if possible, which mean there's no associated container
 docker rmi $image_id
done
```

#### a great removal

```bash
docker system prune -a
```

!!! warning "This command will remove all unused containers, networks, images (both dangling and unreferenced), and optionally, volumes."

## Syncing Changes between Local Directory and Docker Container

When working with projects like FastAPI, Django, ReactJS, MkDocs, NestJs, Flask or similar applications running inside Docker containers, syncing changes between your local development environment and the container becomes crucial for an efficient workflow. This synchronization ensures that any modifications made locally reflect inside the running Docker container, allowing real-time updates without the need for manual file transfers or container restarts.

For instance, when using MkDocs to build documentation or working with FastAPI and Django for web development, syncing changes enables immediate feedback on updates, edits, or additions made to the project files.

You can use the `--mount` option with `type=bind` in Docker accomplishes this synchronization:

```bash
docker run -p 49160:8000 --mount type=bind,source=$DOCS_ABS_PATH,target=/app/docs project_docu
```

This command binds the local `docs` directory (specified by `$DOCS_ABS_PATH`) to the `/app/docs` directory within the running `project_docu` Docker container. Any changes made within the local `docs` directory will be instantly reflected in the container, and vice versa.

This synchronization mechanism ensures that as you modify files within your local development environment—be it Markdown files for MkDocs, source code for FastAPI, or Django—the changes are immediately accessible and reflected within the Docker container. This allows for seamless development, testing, and previewing of changes without interruptions caused by manual file transfers or container restarts.

Integrating this sync approach into your development workflow significantly streamlines the process, enhancing productivity and enabling swift iterations during project development and testing phases.

### Optimizing Container Performance

To maximize the performance and efficiency of your containers, consider the following tips:

- __Keep Images Lean:__ Minimize the size of your Docker images by removing unnecessary dependencies and files.
- __Use Docker Compose:__ Streamline multi-container applications by defining them in a single file with Docker Compose.
- __Monitor Resource Usage:__ Monitor CPU, memory, and disk usage of your containers using Docker stats to identify performance bottlenecks.
- __Implement Container Orchestration:__ Explore container orchestration tools like Kubernetes or Docker Swarm for managing large-scale container deployments.

By incorporating these advanced techniques into your Docker workflow, you can elevate your container management to new heights and unlock the full potential of Docker.

## Conclusion: Embracing the Future of Containerization

Congratulations! You've embarked on a transformative journey through the world of Docker, mastering essential commands, best practices, and advanced techniques along the way. Armed with this knowledge, you're well-equipped to revolutionize your development workflow, streamline deployment processes, and unleash unparalleled efficiency with Docker.

As you continue to explore and experiment with Docker, remember that containerization is not just a technology—it's a mindset. Embrace the principles of consistency, scalability, and efficiency, and let Docker empower you to build, deploy, and scale applications like never before.

## Related links

- [Simple guide to using Docker on Windows 10 and access from WSL 2](./docker.md)
