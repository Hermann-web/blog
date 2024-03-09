---
date: 2023-11-17
authors: [hermann-web]
comments: true
description: >
  A comprehensive guide to setting up a Laravel environment on a windows system,
  covering Apache, PHP, MySQL/MariaDB, Composer, and phpMyAdmin configurations.
categories:
  - frameworks
  - web
  - fullstack
  - laravel
  - windows
  - php
  - mysql
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: Setting Up Laravel Environment on Windows
tags:
  - Apache
  - PHP
  - MySQL
  - Composer
  - phpMyAdmin
---

This guide provides step-by-step instructions to set up a Laravel project on your local environment using XAMPP. If you encounter any issues, please refer to the version details provided below for context and troubleshooting.

## Introduction

Welcome to the "Setting Up Laravel Environment on Windows" tutorial! This guide helps you set up a strong development environment on your Windows system for effortless Laravel web application creation. You'll navigate through configuring Apache, PHP, MySQL/MariaDB, Composer, and phpMyAdmin, making your Windows system a powerful platform for Laravel development. Whether you're new or experienced in web development, this step-by-step tutorial ensures a smooth setup, enabling you to dive into Laravel effortlessly.

In this tutorial, we'll cover:

1. **Installation of Essential Tools**: We'll start by installing XAMPP, Composer, and Node.js. These tools are the building blocks of your Laravel environment.

2. **XAMPP Configuration**: We'll guide you through launching the Apache web server, setting up the MySQL database server, and configuring a database using phpMyAdmin.

<!-- more -->

3. **Database Setup for Laravel**: You'll learn how to configure your Laravel application to interact with the database through the `.env` file.

4. **Dependency Management**: We'll explore installing project dependencies using Composer and Node.js for your Laravel project.

5. **Project Setup and Run**: You'll run essential commands to generate keys, perform database migrations, seed initial data, and finally, start the Laravel development server.

6. **Bonus - Laravel Pattern Overview**: We'll provide a brief comparison of Laravel's architecture to the MVC pattern, similar to Django's structure.

Each section is designed to streamline your Laravel setup process, ensuring you have a robust environment ready to build dynamic web applications. Let's dive in and set up your Windows system for Laravel development!

## Version Details

> from xammp installation

- **PHP Version:** PHP 8.2.4 (cli) (built: Mar 14 2023)
- **XAMPP Control Panel Version:** XAMPP for Windows 8.2.4
- **MySQL Version:** MariaDB 10.4.28, for Win64 (AMD64)

> from composer intallation

- **Composer Version:** Composer version 2.6.2 (2023-09-03)

> from node intallation

- **Node Version:** Node version v20.5.1 (2023-09-03)

## Installation Steps

### Install XAMPP

XAMPP is a free and open-source cross-platform web server solution stack package developed by Apache Friends, consisting mainly of the Apache HTTP Server, MariaDB database, and interpreters for scripts written in the PHP and Perl programming languages.

- Download from the website: [XAMPP Download Page](https://www.apachefriends.org/download.html)
- Put the folder `C:\xampp\php` (or equivalent) in the variables environment

### Install Composer (PHP Dependency Manager)

Composer is a tool for dependency management in PHP. It allows you to declare the libraries your project depends on and it will manage them for you.

```bash
# Download the Composer installer for Windows: link found at https://getcomposer.org/doc/00-intro.md
curl -O https://getcomposer.org/Composer-Setup.exe

# Run the Composer installer (this will open a GUI installer)
start Composer-Setup.exe
```

### Install Node.js and npm (Node Package Manager)

Node.js is an open-source, cross-platform, JavaScript runtime environment that executes JavaScript code outside a web browser. npm is the default package manager for Node.js.

- Download the Node.js (`node 20` preferably) installer for Windows from the official website: <https://nodejs.org/>
- Run the Node.js installer (this will also install npm).

## Setup a laravel project: case of lavsms

### Run XAMPP

XAMPP is used to provide a local server environment to run your Laravel application.

- Start the XAMPP GUI application.
- Launch the Apache web server and MySQL database server.
- Create a new database named `lavsms` using phpMyAdmin or another MySQL client.

    ```bash
    mysql -u your_username -p -e "CREATE DATABASE lavsms;"
    ```

### Database Configuration

Configure the database settings for your Laravel application.

- Create an environment (`.env`) file by making a copy of the example file:

    ```bash
    cp .env.example .env
    ```

- Modify the database connection settings in the `.env` file to match your XAMPP setup:

    ```dotenv
    DB_DATABASE=lavsms
    DB_USERNAME=root
    DB_PASSWORD=
    ```

### Install Project Dependencies

Install the necessary dependencies for your Laravel project.

```bash
# Navigate to the project directory
cd path/to/project

# Update Composer dependencies (if needed)
composer update

# Install Composer dependencies
composer install

# Install Node.js dependencies
npm install
```

### Build

Perform necessary build steps for your Laravel application.

```bash
# Generate an application key
php artisan key:generate

# Clear the configuration cache
php artisan config:clear

# create a symbolic link (`/public/storage`) from the storage directory (`/storage/app/public`) to the public directory (`/public/`)
php artisan storage:link
```

### Build Db

Prepare and set up your database for the Laravel application.

```bash
# Run database migrations to create database tables
php artisan migrate

# Seed the database with initial data (if needed)
php artisan db:seed   
```

### Run Development

Start the development server for your Laravel application.

```bash
# Start the Laravel development server
php artisan serve
```

## (Bonus) A Comparison: Laravel (PHP) vs. Django (Python) MVC-like Architecture
>
> A brief comparison of Laravel's architecture to Django's MVC pattern.

Both Laravel (PHP) and Django (Python) frameworks use a MVC-like architecture. Here are the analogies.

=== ":octicons-file-code-16: `Laravel (PHP)`"

    1. **Routes**: Defined in `routes/web.php`.
        - Invokes PHP controllers.
        - Calls `resources\views\partials\js\custom_js.blade.php` (JavaScript) on form submission, writing to the console.
    1. **Serializers**: Located in `app/http/requests`.
        - Used in controllers for data validation.
    1. **Controllers**: Found in `app/http/controllers`.
        - Utilizes serializers automatically for data validation.
        - Uses models for CRUD operations.
        - Returns `parse(a_view, data_for_client)` similar to Django.
    1. **Models**: Reside in `app/models`.
        - Utilized in controllers for CRUD operations.
    1. **Views (Blade)**: Located in `resources/views`.
        - Similar to PHP-client in Django, handling the presentation layer.

=== ":octicons-file-code-16: `Django (Python)`"

    1. **URL Patterns**: Defined in `urls.py`.
        - Maps to Python views.
        - Handles HTTP requests and defines the view functions.
    1. **Serializers**: Often part of Django REST framework in Python.
        - Used for serialization and deserialization of data.
    1. **Views**: Python files corresponding to the application's logic.
        - Utilizes serializers for data validation.
        - Performs database operations and returns rendered templates.
    1. **Models**: Represented as Python classes in `models.py`.
        - Represents the application's data structure.
        - Interacts with the database via Django's ORM.
    1. **Templates**: HTML files residing in `templates` directory.
        - Renders the user interface based on data provided by views.

## Related pages

- [Guide to Installing MySQL and Connecting to Databases](../../../software-and-tools/sgbd/mysql/comprehensive-guide-to-installing-mysql-and-connecting-to-databases.md)
- [Setting Up Laravel Environment on Linux](./setup-laravel-environment-for-linux.md)
