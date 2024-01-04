---
date: 2023-12-14
authors: [hermann-web]
comments: true
description: |
  A comprehensive guide for beginners diving into the realm of SSH connections and file transfers using Git Bash. From initiating connections to troubleshooting common errors and optimizing file transfers, this guide aims to empower users with the know-how of secure and efficient SSH usage.
categories:
  - devops
  - remote-access
  - ssh
title: "Mastering SSH and File Transfers to Remote servers: A Beginner's Handbook"
---

## Introduction

__Do you find yourself baffled by the intricacies of SSH connections and file transfers to remote servers ?__

Navigating the landscape of SSH connections, troubleshooting connection issues, and securely transferring files across servers can be a daunting task, especially for newcomers. 

<div class="float-img-container float-img-right">
  <a title="Chenyijia001, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Ssh_cyj.png">
    <img style="width: 15rem;" alt="Ssh cyj" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Ssh_cyj.png/512px-Ssh_cyj.png">
  </a>
</div>

This guide is your compass in the world of SSH, unraveling the complexities and providing step-by-step instructions for establishing secure connections and transferring files seamlessly using Git Bash or WSL2 for Windows users and straightforward methods for Linux enthusiasts.


<!-- more -->

Whether you're a developer, sysadmin, or tech enthusiast stepping into the realm of remote server access or seeking efficient file transfer solutions, this guide is tailored to demystify SSH, troubleshoot common pitfalls, and equip you with the skills to maneuver through file transfers effortlessly.

This document break down the process of connecting via SSH and file transfer and should help someone new to SSH understand the process, troubleshoot common issues, and handle file transfers easily

## Requirements:

- For windows users, use Git Bash installed on your computer or use wsl2 (for windows >=10)
- For linux users, this should be straighforward


## Connecting via SSH from cli

### Steps to Connect:
1. **Open Terminal**: Or search for Git Bash in your applications and open it.
2. **Accessing the Server**:
   - Use the command 
      ```bash
      ssh {username}@{domain}
      ``` 
      or 
      ```bash
      ssh {username}@{server_ip}
      ```
   - Replace `{username}` with your remote server username.
   - Replace `{domain}` with the domain name or `{server_ip}` with the server's IP address.
3. **Adding a Specific Port**:
   - If the server uses a different port (usually 22), use `ssh {username}@{domain} -p {port}`. Replace `{port}` with the correct port number.

4. **Entering Credentials**:
   - After executing the command, you'll be prompted to enter your remote server account password. Type it in and press Enter.

## Troubleshooting SSH Connection Issues

### Error: "Unable to negotiate with... port..."
- If you encounter the error `Unable to negotiate with <IP> port <Port>: no matching host key type found. Their offer: ssh-rsa,ssh-dss`
- Solution:
   - Configure the client to accept the host key sent by the server.
   - Edit the `~/.ssh/config` file:
   ```
   Host {domain}
       HostKeyAlgorithms +ssh-rsa,ssh-dss
   ```
   - Use a text editor like Nano, Vim, or Notepad to modify the file.

- Other Common SSH errors:
   - Permissions: Ensure correct file permissions for `~/.ssh` and authorized_keys.
   - Network problems: Check firewall settings and network connectivity.

## Alternative to ssh access though a cli: Putty and OpenSSH

- **Putty (Windows):** 

Known for its user-friendly GUI, Putty offers a straightforward interface for establishing SSH connections on Windows systems. It's particularly popular among users who prefer a graphical interface for SSH connections.

That's why Putty is a popular SSH client primarily used on Windows systems. However, it's worth noting that while Putty is predominantly associated with Windows, it can also be utilized on other operating systems through compatibility layers or third-party tools like Wine on Linux or macOS.

- **OpenSSH:** 

OpenSSH, on the other hand, is an open-source implementation of the SSH protocol. It's available not just for Windows but also for Linux, macOS, and various Unix-like operating systems. OpenSSH provides both the client (ssh) and server (sshd) components, making it a versatile and widely adopted solution for secure remote access, file transfer, and tunneling across different platforms. It offers a robust set of features, including secure remote access, file transfer (using tools like `scp` and `sftp`), and tunneling capabilities.

Thats's why OpenSSH is often preferred by users who work in mixed environments or want a consistent SSH experience across different operating systems. It's commonly used in command-line environments and scripts due to its versatile nature.

## File Transfer Using SSH
You can use `scp` command to transfer files directly between two servers (local to remote or one remote to another) by specifying their addresses and file paths

### Sending Files to Remote Server:

- Use the `scp` command:
   ```bash
   local_file="/path/to/local/file"
   remote_file="$remote_user@$remote_host:/path/to/remote/file/or/folder"
   scp "$local_file" "$remote_file"
   ```
- You'll be prompted for the password before the file is sent.

### Transferring Between Servers:
Similar to local to server transfer, use `scp` between two remote servers by specifying their addresses.

- Use the `scp` command to transfer files directly between two remote servers:
   ```bash
   remote_file_source="$remote_user1@$remote_host1:/path/to/source/file"
   remote_file_destination="$remote_user2@$remote_host2:/path/to/destination/folder"
   scp "$remote_file_source" "$remote_file_destination"
   ```
- Replace:
   - `$remote_user1` with the username for the first remote server.
   - `$remote_host1` with the address or IP of the first remote server.
   - `$remote_user2` with the username for the second remote server.
   - `$remote_host2` with the address or IP of the second remote server.

- This will transfer the specified file from the first remote server to the specified folder on the second remote server.

### SCP Options:

Using `-r` for recursive copying of directories:
```bash
scp -r local_directory username@remote_host:/remote_directory
```

### Alternative File Transfer Methods:

Using `rsync` for efficient synchronization:
```bash
rsync -avz --progress /path/to/source username@remote_host:/path/to/destination
```

## Running commands on a remote server without accessing its cli
For example, using ssh to Fetch latest changes from the remote repository
```bash
ssh "$remote_user@$remote_host" "cd $remote_path && git fetch"
```


## Simplifying SSH Access with sshpass
To avoid being prompted to write the password, `sshpass` is a tool you want

- **Install sshpass**:
   - If needed, install sshpass using `sudo apt install sshpass`.
- **Accessing SSH without Password Prompt**:
   Instead of
   ```bash
   ssh "$remote_user@$remote_host"
   ```
   use
   ```bash
   sshpass -p "$password" ssh "$remote_user@$remote_host"
   ```
- Example: If want to tetch latest changes from the remote repository, i run
```bash
sshpass -p "$password" ssh "$remote_user@$remote_host" "cd $remote_path && git fetch"
```




## Conclusion

__Congratulations! You've now mastered the fundamentals of SSH connections and file transfers using Git Bash.__

In this guide, we've covered the essential steps to initiate SSH connections, troubleshoot common errors, and conduct seamless file transfers between local and remote servers. You've learned to troubleshoot connection issues, enhance security configurations, and optimize file transfers using `scp` and `rsync`.

Remember, SSH is a powerful tool for secure communication and file transfer, and understanding its nuances empowers you to work efficiently across different servers and systems.

As you continue your journey, keep exploring the capabilities of SSH, experiment with different options and configurations, and don't hesitate to delve deeper into security best practices.

Whether you're a developer collaborating on remote repositories or a system administrator managing servers, the knowledge gained here will serve as a solid foundation for your endeavors.

Embrace the power of SSH, continue to explore, and may your future endeavors in remote access and file transfer be secure, efficient, and hassle-free!
