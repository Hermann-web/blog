---
date: 2023-11-12
authors: [hermann-web]
comments: true
description: >
  This blog shows useful stuff to know about or learn to do as web developer or data scientist/engineer
  Always working the fastest and easiest ways toward success
categories:
  - database-management
  - mysql
  - linux
  - deployment
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "Guide to Installing MySQL and Connecting to Databases"
---


## Introduction

MySQL is a popular relational database management system used for storing and managing data. To get started, you'll need to install MySQL, set it up, and then connect to databases. Here's a comprehensive guide to help you through the process.

## Installation Process

To install MySQL, follow these steps:

- update the package lists

```bash
sudo apt-get update
```

- install MySQL

```bash
sudo apt-get install mysql-server
```

<!-- more -->

or use yum

```bash
yum install <package-name>
or 
sudo yum install mysql-server
```

you will be prompted to set a password for the MySQL root user. Make sure to choose a strong password and remember it, as you will need it to access MySQL.

- if you haven't been prompted the password,
  - do this

    ```bash
    sudo mysql_secure_installation
    ```

  - or connect later on using your ssh details (username, password): ```mysql -u [username] -p```

- start the MySQL service

```bash
sudo service mysql start
or
sudo systemctl start mysqld
```

- check if MySQL is running,

```bash
sudo service mysql status
```

If MySQL is running, you should see a message that says "Active: active (running)".

## Testing the MySQL Connection

Here are a few commands you can use to test your MySQL connection:

```shell
mysql -u [username] -p -h [hostname] -P [port]
```

- Replace [username] with your database username, [hostname] with your database hostname, [port] with your database port number, and leave out the brackets.

- For example, if your database username is "myuser", your database hostname is "db.example.com", and your database port number is 3306, the command would look like this:

```shell
mysql -u myuser -p -h db.example.com -P 3306
```

- Press Enter and then enter your database password when prompted. If the connection is successful, you'll see a prompt that looks like this:

- if that don't work, test that command with your ssh [username] and [password]

```shell
mysql -u [username] -p
```

you will see this

```shell
mysql>
```

This means you're now connected to your MySQL server.

- To test that you can retrieve data from your database, enter the following command:

```shell
use [databasename];
select * from [tablename];
```

- Replace [databasename] with the name of your database and [tablename] with the name of a table in your database. This will select all rows from the specified table.

- If the command returns data from your database, then your connection is working properly.

### Essential Actions

Here are some quick commands and actions you can perform in MySQL:

- show all the databases

```shell
SHOW DATABASES;
```

- create a new database

```shell
CREATE DATABASE mydatabase;
```

- use this database:

```shell
USE mydatabase;
```

- create a new table:

```shell
CREATE TABLE mytable (id INT, name VARCHAR(20));
```

- insert data into this table:

```sql
INSERT INTO mytable VALUES (1, 'John'), (2, 'Jane');
```

- query the data:

```sql
SELECT * FROM mytable;
```

This will display the data you inserted into the table.

- To exit the MySQL shell:

```sql
exit
```

### Create a new MySQL user account

```bash
sudo mysql -u root -p
```

```sql
CREATE USER 'yourusername'@'localhost' IDENTIFIED BY 'yourpassword';

GRANT ALL PRIVILEGES ON *.* TO 'yourusername'@'localhost' WITH GRANT OPTION;

FLUSH PRIVILEGES;
```

Replace yourusername and yourpassword with the desired username and password for your MySQL user account.

### Connecting to an Online MySQL Database

```
# Extracting details from the connection string
username="doadmin"
password="AVNS_7wyTjplB7LVpwf3VKKf"
hostname="db-mysql-metalandapi-do-user-12655475-0.b.db.ondigitalocean.com"
port="25060"
database_name="defaultdb"

# Constructing the MySQL CLI command
mysql -u $username -p$password -h $hostname -P $port $database_name
```

### Deployment and Integration

Configure your web application to utilize the MySQL database. Ensure your web server knows:

- `Hostname` (usually localhost if on the same machine)
- `Port` (default is 3306 for MySQL)
- `Username` and `password` created earlier
Database name you established

Employ programming languages like PHP or Python to interact with the MySQL database. Use Nginx as a reverse proxy to direct requests to your application server.

### Example: Integrating MySQL with Maven

An example of integrating MySQL with a Maven project:

1. Modify your Maven project to connect to the MySQL database. You can add the MySQL JDBC driver as a dependency in your project's `pom.xml` file, like this:

```xml
<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <version>8.0.25</version>
</dependency>
```

Replace the version number with the latest version of the MySQL JDBC driver.

2. Configure your application to use the MySQL database. You can add the necessary configuration properties to your application.properties file, like this:

```properties
spring.datasource.url=jdbc:mysql://localhost:3306/your_database_name
spring.datasource.username=root
spring.datasource.password=your_mysql_password
spring.jpa.hibernate.ddl-auto=update
```

Replace **your_database_name** with the name of the database you created in step 2, and **your_mysql_password** with the password you set for the MySQL root user.

3. Build your Maven project and create an executable JAR file using the mvn package command.

4. Start your application using the executable JAR file you created earlier. You can start it using the ```java -jar <jar-file-name>``` command.

Now your application should be up and running, connected to the MySQL database and loaded with the data from your SQL file. Nginx can then be used as a reverse proxy to serve your application to users.
