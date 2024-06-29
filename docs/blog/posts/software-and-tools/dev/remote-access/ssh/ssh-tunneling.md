---
date: 2024-06-29
authors: [hermann-web]
comments: true
description: >
  A comprehensive guide to SSH tunneling, file transfers, and key-based authentication.
categories:
  - dev
  - remote-access
  - ssh
  - Security
title: "Comprehensive Guide to SSH: Tunneling, File Transfers, and Key-Based Authentication"
---

# Comprehensive Guide to SSH: Tunneling, File Transfers, and Key-Based Authentication

SSH (Secure Shell) is a powerful protocol used for secure communication between computers, offering a wide range of functionalities including secure remote access, file transfers, and key-based authentication.

In this guide, we'll cover various aspects of SSH, including:

- Port tunneling for secure access to services like MySQL.
- Secure file transfer using SFTP and SCP.
- SSH agent forwarding and key-based authentication for enhanced security.
- Dynamic port forwarding (SOCKS proxy) for secure browsing.

<!-- more -->

This document provides detailed instructions and examples to help you harness the full potential of SSH for secure communication and efficient remote server management.

## Port Tunneling

SSH port tunneling securely forwards network traffic from a local machine to a remote server. This is crucial for accessing services like MySQL databases running on remote servers.

### Local Port Forwarding

Local port forwarding forwards a port on your local machine to a port on the remote server.

#### Example Command

To access a MySQL database on a remote server:

```bash
ssh -L <local_port>:<db_host>:<db_port> <SSH_USER>@<SSH_HOST>
```

- `<local_port>`: Local port to forward (e.g., `3306`).
- `<db_host>`: Hostname or IP of the database server (often `localhost`).
- `<db_port>`: Port of the database service (e.g., `3306` for MySQL).
- `<SSH_USER>`: Your SSH username.
- `<SSH_HOST>`: IP address or hostname of the remote server.

#### Example

```bash
ssh -L 3306:localhost:3306 user@remote-server.com
```

This command forwards traffic from `localhost:3306` on your local machine to `localhost:3306` on `remote-server.com`. You can then connect to the remote MySQL database using a local MySQL client.

### Remote Port Forwarding

Remote port forwarding forwards a port from the remote server to a local machine.

#### Example Command

```bash
ssh -R <remote_port>:<local_host>:<local_port> <SSH_USER>@<SSH_HOST>
```

- `<remote_port>`: Port on the remote server to forward.
- `<local_host>`: Hostname or IP of your local machine (often `localhost`).
- `<local_port>`: Port on your local machine.

#### Example

```bash
ssh -R 8080:localhost:80 user@remote-server.com
```

This forwards traffic from `remote-server.com:8080` to `localhost:80` on your local machine.

### MySQL Tunneling Example

Below is an example script to tunnel MySQL traffic using SSH. Ensure MySQL is installed on both the client and remote server and that necessary permissions are set up:

#### Prerequisites (Remote Server)

- MySQL installed and running on the remote server.
- Proper permissions set for SSH access and MySQL user (`dbuser` in this example).

#### Example Script

```bash
# Configuration
SSH_PORT=22
HOST=example.com
USER=username

LOCAL_PORT=5523
REMOTE_DB_HOST=127.0.0.1
REMOTE_DB_PORT=3306

# Establish SSH Tunnel (Run on client machine)
ssh -f ${USER}@${HOST} -p ${SSH_PORT} -L ${LOCAL_PORT}:${REMOTE_DB_HOST}:${REMOTE_DB_PORT} -N

# Connect to MySQL (Run on client machine)
mysql -u dbuser -p -h 127.0.0.1 -P ${LOCAL_PORT}
```

#### Explanation

1. **Variables**:
   - `SSH_PORT`: Port used for SSH connections (default is `22`).
   - `HOST`: Remote server’s address.
   - `USER`: Your SSH username.
   - `LOCAL_PORT`: Local port to forward MySQL traffic (e.g., `5523`).
   - `REMOTE_DB_HOST`: Remote database host (usually `127.0.0.1`).
   - `REMOTE_DB_PORT`: Remote database port (default for MySQL is `3306`).

2. **SSH Command**:
   - `-f`: Runs command in the background.
   - `-L`: Sets up local port forwarding from `LOCAL_PORT` to `REMOTE_DB_HOST:REMOTE_DB_PORT`.
   - `-N`: Prevents execution of remote commands (only sets up the tunnel).

3. **MySQL Connection**:
   - Connects to MySQL using the local tunnel.
   - `-u dbuser`: Specifies MySQL username.
   - `-p`: Prompts for password.
   - `-h 127.0.0.1`: Connects to localhost (tunneled).
   - `-P ${LOCAL_PORT}`: Specifies local port for the tunnel.

#### Usage

1. Ensure MySQL is installed and accessible on both client and remote servers.
2. Run the SSH tunneling script on the client machine to establish the tunnel.
3. Use the MySQL command to connect to the remote database via the local tunnel (`localhost:5523` in this example).

This method securely encrypts MySQL traffic, maintaining data privacy during transmission. Adjust ports and credentials as per your specific setup.

## Secure File Transfer Protocol (SFTP)

SFTP [^sftp-wiki] is a secure way to transfer files between your local machine and a remote server using SSH. It encrypts both commands and data, providing a high level of security.

### Steps to Use SFTP

1. **Connect to the Remote Server**

   Use the following command to start an SFTP session:

   ```bash
   sftp <SSH_USER>@<SSH_HOST>
   ```

   - Replace `<SSH_USER>` with your SSH username.
   - Replace `<SSH_HOST>` with your remote server's IP address or hostname.

2. **Common SFTP Commands**

   - `ls`: List files on the remote server.
   - `cd <directory>`: Change directory on the remote server.
   - `get <remote_file>`: Download a file from the remote server.
   - `put <local_file>`: Upload a file to the remote server.
   - `exit`: Close the SFTP session.

### Example Usage

```bash
sftp username@example.com
sftp> ls
sftp> cd /path/to/directory
sftp> get remote_file.txt
sftp> put local_file.txt
sftp> exit
```

## Secure Copy (SCP)

SCP [^scp-openbsd] allows you to securely transfer files between hosts using SSH. It's a straightforward way to copy files securely.

### Example Commands

- **Copy a file from local to remote:**

  ```bash
  scp <local_file> <SSH_USER>@<SSH_HOST>:<remote_path>
  ```

- **Copy a file from remote to local:**

  ```bash
  scp <SSH_USER>@<SSH_HOST>:<remote_file> <local_path>
  ```

## SSH Agent Forwarding

SSH agent forwarding [^ssh-agent-forwading] allows you to use your local SSH keys on remote servers, enabling seamless access to additional remote servers without copying keys.

### Usage

```bash
ssh -A <SSH_USER>@<SSH_HOST>
```

## Dynamic Port Forwarding (SOCKS Proxy)

Using SSH, you can create a SOCKS proxy [^dynamic-port-forwarding] that routes traffic from applications through the SSH tunnel, allowing secure browsing.

### Command

```bash
ssh -D <local_port> <SSH_USER>@<SSH_HOST>
```

## SSH Key-Based Authentication

For enhanced security, use SSH keys instead of passwords for authentication [^ssh-key-auth]. This prevents unauthorized access and simplifies the login process.

### Steps to Set Up

1. **Generate an SSH Key Pair**

   ```bash
   ssh-keygen -t rsa -b 4096 -C "<email@example.com>"
   ```

2. **Copy the Public Key to the Remote Server**

   ```bash
   ssh-copy-id <SSH_USER>@<SSH_HOST>
   ```

This setup allows you to log in securely without entering a password.

### Example: SSH Key-Based Authentication with GitHub

Setting up SSH key-based authentication with GitHub [^connecting-to-github-with-ssh] enhances security while maintaining ease of use for your development workflows. Recently, GitHub deprecated the use of RSA keys with SHA-1 due to security concerns, requiring users to switch to more secure algorithms like `ed25519` [^improving-git-protocol-security-github].

#### Error Message

If you encounter the following error:

```plaintext
ERROR: You're using an RSA key with SHA-1, which is no longer allowed.
Please use a newer client or a different key type.
```

You need to switch to `ed25519`:

#### Steps to Set Up SSH Key-Based Authentication

1. **Generate an SSH Key Pair**

   Use `ssh-keygen` to generate a new SSH key pair with the Ed25519 algorithm:

   ```bash
   ssh-keygen -t ed25519 -C "your-email-address"
   ```

   Follow the prompts to save the key pair in the default location (`~/.ssh/id_ed25519`).

2. **Start the SSH Agent**

   To manage your SSH keys, start the SSH agent:

   ```bash
   eval "$(ssh-agent -s)"
   ```

3. **Add the SSH Private Key to the SSH Agent**

   Add your SSH private key to the SSH agent:

   ```bash
   ssh-add ~/.ssh/id_ed25519
   ```

4. **Copy the SSH Public Key to GitHub**

   Retrieve your SSH public key and copy its contents:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

   Copy the entire output.

5. **Add the SSH Key to GitHub**

   - Go to [GitHub Settings > SSH and GPG keys](https://github.com/settings/keys).
   - Click on "New SSH key" or "Add SSH key", paste your SSH public key, and give it a descriptive title.

6. **Verify SSH Connection to GitHub**

   Test your SSH connection to GitHub:

   ```bash
   ssh -T git@github.com
   ```

   You should see a message indicating successful authentication.

#### Example: Clone a GitHub Repository Using SSH

To clone a repository from GitHub:

```bash
git clone git@github.com:your-username/your-repo.git
```

Replace `your-username` and `your-repo` with your GitHub username and repository name.

### Configuration Update for `ssh-ed25519`

If your system does not support `ed25519` by default, update your SSH configuration file (`~/.ssh/config`) to include it.

For example, i once changed from:

```plaintext
HostKeyAlgorithms ssh-rsa,rsa-sha2-512
PubkeyAcceptedKeyTypes ssh-rsa
```

to:

```plaintext
HostKeyAlgorithms ssh-rsa,rsa-sha2-512,ssh-ed25519
PubkeyAcceptedKeyTypes ssh-rsa,ssh-ed25519
```

This ensures compatibility with `ed25519` keys, providing enhanced security for your connections.

### Conclusion

SSH is a versatile tool that offers secure communication, file transfers, and more. By using SSH, you can protect your data and manage remote servers efficiently, especially with updated algorithms like `ed25519` ensuring enhanced security.

[^sftp-wiki]: [SSH File Transfer Protocol (SFTP)](https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol)
[^scp-openbsd]: [Secure Copy Protocol (SCP)](https://man.openbsd.org/scp)
[^ssh-agent-forwading]: [SSH Agent Forwarding](https://www.ssh.com/academy/ssh/agent#ssh-agent-forwarding)
[^dynamic-port-forwarding]: [SOCKS Proxy (Dynamic Port Forwarding)](https://www.ssh.com/academy/ssh/tunneling#dynamic-port-forwarding-dynamic-ssh-tunneling)
[^ssh-key-auth]: [SSH Key Authentication](https://www.ssh.com/academy/ssh/key)
[^connecting-to-github-with-ssh]: [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
[^improving-git-protocol-security-github]: [Improving Git Protocol Security on GitHub](https://github.blog/2021-09-01-improving-git-protocol-security-github/)
