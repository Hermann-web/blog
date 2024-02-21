---
date: 2023-11-18
authors: [hermann-web]
comments: true
description: >
  A comprehensive guide to setting up a Laravel environment on an Ubuntu system,
  covering Apache, PHP, MySQL/MariaDB, Composer, and phpMyAdmin configurations.
categories:
  - frameworks
  - web
  - fullstack
  - laravel
  - linux
  - deployment
  - php
  - mysql
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: Setting Up Laravel Environment on Ubuntu
tags:
  - Apache
  - PHP
  - MySQL
  - Composer
  - phpMyAdmin
---


This guide will help you set up the necessary environment to run a Laravel application on an Ubuntu system.

So this document provides a step-by-step guide to set up Apache, PHP, MySQL/MariaDB, Composer, and phpMyAdmin for managing databases, while also ensuring MySQL root user password setup for a Laravel environment on Ubuntu.

## Install Apache and PHP

- Install Apache

```bash
sudo apt update 
sudo apt install apache2
```

<!-- more -->

- Enable the Apache service and start it:

```bash
sudo systemctl enable apache2
sudo systemctl start apache2
```

- Install php

```bash
sudo apt install php php-cli php-common php-mbstring php-xml php-zip php-mysql php-pgsql php-sqlite3 php-json php-bcmath php-gd php-tokenizer php-xmlwriter
```

Check `localhost` in your browser to ensure Apache is running.

## Install MariaDB/MySQL

- Install MariaDB
[MariaDB](https://kinsta.com/blog/mariadb-python/) is an open-source relational database management system. Install it by running the following command:

```bash
sudo apt install mariadb-server
```

- Install MySQL

```bash
sudo mysql_secure_installation
```

- Enable and start MySQL:

```bash
sudo systemctl enable mysql
sudo systemctl start mysql
sudo systemctl status mysql
```

## Install Composer

Composer is a dependency management tool for PHP. Install with the command below

```bash
sudo apt install composer
```

- here is an altenative

```bash
curl -sS https://getcomposer.org/installer | php
sudo mv composer.phar /usr/local/bin/composer
sudo chmod +x /usr/local/bin/composer
```

## Install phpMyAdmin for Database Management

```bash
sudo apt update
sudo apt install phpmyadmin
sudo ln -s /etc/phpmyadmin/apache.conf /etc/apache2/conf-available/phpmyadmin.conf
sudo a2enconf phpmyadmin
sudo systemctl reload apache2
```

Access phpMyAdmin at `http://localhost/phpmyadmin` to manage your databases.

## Configure MySQL Root User Password
>
> because phpmyadmin refuse the passwordless login and also for security purposes

Modify the MySQL configuration file:

```bash
sudo nano /etc/mysql/mariadb.conf.d/50-server.cnf
```

In some setups or older versions, this file might exist as the main configuration file for the MySQL/MariaDB server

```bash
sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf
```

Add the following line under the `[mysqld]` section:

```conf
skip-grant-tables
```

Restart MySQL in safe mode to update the config:

```bash
sudo systemctl stop mysql
sudo mysqld_safe --skip-grant-tables --skip-networking &
```

If there's a process conflict, kill the processes and restart MySQL.

In another terminal:

```sql
use mysql;
update user set authentication_string=PASSWORD("new_password") where User='root';
flush privileges;
quit;
```

Restart MySQL:

```bash
sudo systemctl stop mysql
sudo systemctl start mysql
```

Access phpMyAdmin at `http://localhost/phpmyadmin` and use the updated root password.

This environment should now be ready for running your Laravel application.

## (Bonus) Create a New Laravel Project

Let's conclude this guide with a practical example by demonstrating how to create a new Laravel project using Composer.

So we will apply the newly configured environment by creating a new Laravel project and accessing it through a web browser

Now that you have set up your Laravel environment, let's create a new Laravel project using Composer.

Run the following command in your terminal:

```bash
composer create-project --prefer-dist laravel/laravel my-laravel-app
```

This command will create a new Laravel project named `my-laravel-app` in the current directory. Replace `my-laravel-app` with your preferred project name.

Navigate to the project directory:

```bash
cd my-laravel-app
```

Then, start the Laravel development server:

```bash
php artisan serve
```

Access your Laravel application by visiting `http://localhost:8000` in your web browser. You should see the default Laravel welcome page, confirming that your new Laravel project is up and running!

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

### Laravel (PHP)

- **Routes**: Defined in `routes/web.php`.
  - Invokes PHP controllers.
  - Calls `resources\views\partials\js\custom_js.blade.php` (JavaScript) on form submission, writing to the console.
- **Serializers**: Located in `app/http/requests`.
  - Used in controllers for data validation.
- **Controllers**: Found in `app/http/controllers`.
  - Utilizes serializers automatically for data validation.
  - Uses models for CRUD operations.
  - Returns `parse(a_view, data_for_client)` similar to Django.
- **Models**: Reside in `app/models`.
  - Utilized in controllers for CRUD operations.
- **Views (Blade)**: Located in `resources/views`.
  - Similar to PHP-client in Django, handling the presentation layer.

### Django (Python)

- **URL Patterns**: Defined in `urls.py`.
  - Maps to Python views.
  - Handles HTTP requests and defines the view functions.
- **Serializers**: Often part of Django REST framework in Python.
  - Used for serialization and deserialization of data.
- **Views**: Python files corresponding to the application's logic.
  - Utilizes serializers for data validation.
  - Performs database operations and returns rendered templates.
- **Models**: Represented as Python classes in `models.py`.
  - Represents the application's data structure.
  - Interacts with the database via Django's ORM.
- **Templates**: HTML files residing in `templates` directory.
  - Renders the user interface based on data provided by views.
