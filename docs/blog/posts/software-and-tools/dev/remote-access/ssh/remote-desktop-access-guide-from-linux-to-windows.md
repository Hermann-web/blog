---
date: 2024-03-11
authors: [hermann-web]
comments: true
description: >
  A comprehensive guide to setting up remote desktop access from a Linux system to a Windows 10 machine using Remmina.
categories:
  - linux
  - RDP
  - windows
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: Setting Up Remote Desktop Access with Remmina on Ubuntu
---

# How to Set Up Remote Desktop Access from Linux to Windows 10 Using Remmina

Remote desktop access has become an essential feature in today's digital landscape, allowing users to connect to their computers from anywhere. While Windows users have built-in options for remote desktop access, Linux users often need to rely on third-party applications. In this guide, we'll explore how to set up remote desktop access from a Linux system to a Windows 10 machine using Remmina.

<!-- more -->

## Microsoft's Remote Desktop Services

Microsoft offers [extensive documentation on remote desktop services](https://learn.microsoft.com/en-us/windows-server/remote/remote-desktop-services/clients/remote-desktop-clients), providing official clients for various platforms such as Windows 10, macOS, and others. However, there isn't an official client for Linux systems. This gap has led Linux users to explore alternative solutions, with Remmina being a popular choice.

## Step 1: Preparing Your Windows 10 Machine

Before connecting remotely to your Windows 10 machine, you'll need to enable remote connections.

1. **Enable Remote Connections**: Navigate to the "Remote Desktop settings" on your Windows 10 machine and ensure that remote connections are allowed.

2. **Find the IP Address**: While the computer name is usually used for remote connections, you can also use the private IP address of the Windows machine. You can find this IP address by running `ipconfig` in the Command Prompt and copying the IPv4 address listed under "Carte réseau sans fil Wi-Fi > Adresse IPv4".

3. **Add Remote Desktop Account**: Add the account (Windows session) you plan to use for remote access. While administrator accounts should work, it's recommended to use a specific user account for a smoother experience.

## Step 2: Installing and Configuring Remmina on Linux

Remmina is an open-source remote desktop client for Linux systems, offering an intuitive interface and robust features.

1. **Install Remmina**: Open a terminal on your Linux system and install Remmina using your package manager:

   ```bash
   sudo apt install remmina
   ```

2. **Create a New Connection**: Launch Remmina and create a new connection profile. Enter the private IP address of your Windows 10 machine as the server address, and provide the username and password of the session you added in the previous step.

3. **Establish the Connection**: Once you've entered the necessary information, click "Connect" to establish the remote desktop connection to your Windows 10 machine.

## Step 3: Testing Remote Desktop Access

Now that you've set up the connection, it's time to test remote desktop access.

1. **Within the Same Network**: Connect to your Windows 10 machine from your Linux system while both devices are on the same network. This allows you to ensure that everything is set up correctly.

2. **Outside the Network (Optional)**: If you encounter issues connecting from outside the network, it may be due to network restrictions or firewall settings. In such cases, you may need to contact your network administrator to allow remote desktop connections from external locations. Alternatively, consider using a virtual private network (VPN) to establish a secure connection to your network and access the Windows 10 machine remotely.

By testing remote desktop access within and potentially outside the network, you can verify the functionality of your setup and troubleshoot any connectivity issues effectively.

By following these steps, you can enjoy seamless remote desktop access from your Linux system to a Windows 10 machine, enhancing your productivity and flexibility.

## Conclusion

Setting up remote desktop access from a Linux system to a Windows 10 machine using Remmina offers convenience and flexibility, allowing users to access their computers remotely from anywhere. While the process involves a few initial setup steps and potential network considerations, the ability to connect seamlessly enhances productivity and enables efficient remote work. By following the steps outlined in this guide and troubleshooting any connectivity issues, users can enjoy the benefits of remote desktop access with ease.
