---
date: 2024-06-16
authors: [hermann-web]
comments: true
description: >
  Explore command-line interface (CLI) syntax equivalence across MySQL, PostgreSQL, and MongoDB for effective cross-platform database management.
categories:
  - database-management
  - cli-tools
  - tools-comparison
  - mysql
  - postgresql
  - mongodb
links:
  - blog/posts/a-guide-to-sql-queries.md
  - blog/posts/mongodb-cli-tips.md
title: "Database Management CLI: Equivalence in MySQL, PostgreSQL, and MongoDB"
---

# Database Management CLI: Equivalence in MySQL, PostgreSQL, and MongoDB

## Introduction

Database management and querying are critical tasks for developers and database administrators. This guide explores syntax equivalences in MySQL, PostgreSQL, and MongoDB, enabling you to transition seamlessly between these systems using their command-line interfaces (CLI).

Understanding the corresponding syntaxes in each database system facilitates code portability and collaboration among developers and administrators across different platforms.

<!-- more -->

## Key Considerations

### Choosing the Right Database System

- **Query Language:** SQL for relational databases (MySQL, PostgreSQL) and MongoDB Query Language (MQL) for NoSQL.
- **Use Case:** Transactional applications, data analytics, or document storage.
- **Scalability:** Horizontal vs. vertical scaling.
- **Community Support:** Size and activity of the user community.
- **Performance:** Performance requirements for read and write operations.

### Databases Overview

=== ":octicons-database-16: MySQL"

    - **CLI Tool:** `mysql`
    - **Query Language:** SQL
    - **Syntax Highlights:** Standard SQL with some MySQL-specific extensions.

=== ":octicons-database-16: PostgreSQL"

    - **CLI Tool:** `psql`
    - **Query Language:** SQL
    - **Syntax Highlights:** Advanced SQL features, support for JSON, and extensive indexing options.

=== ":octicons-database-16: MongoDB"

    - **CLI Tool:** `mongosh`
    - **Query Language:** MongoDB Query Language (MQL)
    - **Syntax Highlights:** Document-based queries, flexible schema, and aggregation framework.

## Comparison Tables

### Connection and Basic Commands

| Task                         | MySQL CLI (`mysql`)           | PostgreSQL CLI (`psql`)               | MongoDB CLI (`mongosh`)               |
| ---------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------- |
| Connect to Database          | `mysql -u username -p database` | `psql -U username -d database`      | `mongosh "mongodb://username:password@host:port/database"` |
| List Databases               | `SHOW DATABASES;`             | `\l`                                 | `show dbs`                            |
| Select Database              | `USE database;`               | `\c database`                        | `use database`                        |
| List Collections/Tables      | `SHOW TABLES;`                | `\dt`                                | `show collections`                    |
| Exit CLI                     | `exit` or `\q`                | `\q`                                 | `exit`                                |

### Database Management

| Task                         | MySQL CLI (`mysql`)           | PostgreSQL CLI (`psql`)               | MongoDB CLI (`mongosh`)               |
|------------------------------|-------------------------------|---------------------------------------|---------------------------------------|
| Create Database              | `CREATE DATABASE dbname;`     | `CREATE DATABASE dbname;`             | `use dbname` (created on first write) |
| Delete Database              | `DROP DATABASE dbname;`       | `DROP DATABASE dbname;`               | `use dbname; db.dropDatabase()`       |

### Table/Collection Management

| Task                         | MySQL CLI (`mysql`)           | PostgreSQL CLI (`psql`)               | MongoDB CLI (`mongosh`)               |
| ---------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------- |
| Create Table/Collection      | `CREATE TABLE table_name (...);` | `CREATE TABLE table_name (...);`     | `db.createCollection("collection_name")` |
| Drop Table/Collection        | `DROP TABLE table_name;`      | `DROP TABLE table_name;`              | `db.collection_name.drop()`           |
| Describe Table/Collection    | `DESCRIBE table_name;`        | `\d table_name`                       | `db.collection_name.stats()`          |
| Rename Table/Collection      | `RENAME TABLE old_name TO new_name;` | `ALTER TABLE old_name RENAME TO new_name;` | `db.collection_name.renameCollection("new_name")` |

### Data Manipulation

| Task                         | MySQL CLI (`mysql`)           | PostgreSQL CLI (`psql`)               | MongoDB CLI (`mongosh`)               |
| ---------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------- |
| Insert Data                  | `INSERT INTO table_name (...) VALUES (...);` | `INSERT INTO table_name (...) VALUES (...);` | `db.collection_name.insertOne({...})` |
| Select Data                  | `SELECT * FROM table_name;`   | `SELECT * FROM table_name;`           | `db.collection_name.find({})`         |
| Update Data                  | `UPDATE table_name SET ... WHERE ...;` | `UPDATE table_name SET ... WHERE ...;` | `db.collection_name.updateOne({...}, {$set: {...}})` |
| Delete Data                  | `DELETE FROM table_name WHERE ...;` | `DELETE FROM table_name WHERE ...;`   | `db.collection_name.deleteOne({...})` |

### Querying Data

| Task                         | MySQL CLI (`mysql`)           | PostgreSQL CLI (`psql`)               | MongoDB CLI (`mongosh`)               |
| ---------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------- |
| Basic Select                 | `SELECT * FROM table_name;`   | `SELECT * FROM table_name;`           | `db.collection_name.find({})`         |
| Where Clause                 | `SELECT * FROM table_name WHERE condition;` | `SELECT * FROM table_name WHERE condition;` | `db.collection_name.find({condition})` |
| Join Tables                  | `SELECT * FROM table1 JOIN table2 ON condition;` | `SELECT * FROM table1 JOIN table2 ON condition;` | `db.collection1.aggregate([{$lookup: {from: "collection2", localField: "field1", foreignField: "field2", as: "joined_docs"}}])` |
| Group By                     | `SELECT column, COUNT(*) FROM table_name GROUP BY column;` | `SELECT column, COUNT(*) FROM table_name GROUP BY column;` | `db.collection_name.aggregate([{$group: {_id: "$column", count: {$sum: 1}}}])` |

### Index Management

| Task                         | MySQL CLI (`mysql`)           | PostgreSQL CLI (`psql`)               | MongoDB CLI (`mongosh`)               |
| ---------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------- |
| Create Index                 | `CREATE INDEX idx_name ON table_name(column);` | `CREATE INDEX idx_name ON table_name(column);` | `db.collection_name.createIndex({column: 1})` |
| List Indexes                 | `SHOW INDEX FROM table_name;` | `\di table_name`                      | `db.collection_name.getIndexes()`     |
| Drop Index                   | `DROP INDEX idx_name ON table_name;` | `DROP INDEX idx_name;`                | `db.collection_name.dropIndex("idx_name")` |

### Transactions

| Task                         | MySQL CLI (`mysql`)           | PostgreSQL CLI (`psql`)               | MongoDB CLI (`mongosh`)               |
| ---------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------- |
| Begin Transaction            | `START TRANSACTION;`          | `BEGIN;`                              | `session = db.getMongo().startSession(); session.startTransaction();` |
| Commit Transaction           | `COMMIT;`                     | `COMMIT;`                             | `session.commitTransaction(); session.endSession();` |
| Rollback Transaction         | `ROLLBACK;`                   | `ROLLBACK;`                           | `session.abortTransaction(); session.endSession();` |

### Import/Export Data

| Task                         | MySQL CLI (`mysql`)           | PostgreSQL CLI (`psql`)               | MongoDB CLI (`mongosh`)               |
| ---------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------- |
| Import Data                  | `LOAD DATA INFILE 'file.csv' INTO TABLE table_name;` | `\COPY table_name FROM 'file.csv' DELIMITER ',' CSV;` | `mongoimport --db database --collection collection_name --file file.json` |
| Export Data                  | `SELECT * FROM table_name INTO OUTFILE 'file.csv';` | `\COPY (SELECT * FROM table_name) TO 'file.csv' DELIMITER ',' CSV;` | `mongoexport --db database --collection collection_name --out file.json` |

## Conclusion

This guide provides a comparison of the most commonly used database management systems' command-line interfaces: MySQL, PostgreSQL, and MongoDB. By understanding these equivalences, developers and administrators can efficiently manage and query databases across different platforms.

Whether you're working with relational databases like MySQL and PostgreSQL or a document-based NoSQL database like MongoDB, having a quick reference for CLI commands can enhance your productivity and streamline your workflow.
