---
date: 2023-11-13
authors: [hermann-web]
description: >
  This blog shows useful stuff to know about or learn to do as web developer or data scientist/engineer
  Always working the fastest and easiest ways toward success
categories:
  - software-and-tools
  - sgbd
  - mongodb
links:
  - setup/setting-up-a-blog.md
  - plugins/blog.md
title: "Guide to Applying query on you mongodb atlas hosted database from command line"
---

## Introduction
Often when you're using a database in a dev project, you want to access it quickly to check for modifications. When you're working with mysql database, you have a [client that can help you with that](../mysql/comprehensive-guide-to-installing-mysql-and-connecting-to-databases.md). But what to do when you're using mongo db ?
In this tutorial, i present how to access, from command line, your databased hosted with mongo db atlas. Then i showcase basic still important query examples.
It should work also for those hosted locally.

## Prerequistes
- Node js
- a mongo db database, offline or locally served

## Download and Install Mongosh

You can download and install `mongosh` from the MongoDB [website](https://www.mongodb.com/try/download/shell) or using package managers like npm or yarn. Make sure you have Node.js installed on your system before proceeding.

Personally, i've downloaded it (and installed the setup) from the [website](https://www.mongodb.com/try/download/shell), put the bin file (containing mongosh.exe) into environment variables and read a bit of the [docu](https://www.mongodb.com/docs/mongodb-shell/) 

Let's use the option 1, as it is a more straightforward approach

=== ":octicons-file-code-16: `Using npm (Node Package Manager)`"

    ```bash
    npm install -g mongodb
    ```

=== ":octicons-file-code-16: `Using yarn (Package Manager)`"

    ```bash
    yarn global add mongodb
    ```

<!-- more -->

For more details on installing `mongosh`, refer to the MongoDB documentation [^mongosh-install].

## Access a dababase from cli
I assume you have a MongoDB deployment to connect to. You can use a free cloud-hosted deployment like MongoDB Atlas or run a local MongoDB deployment.

Connect to your MongoDB deployment using mongosh by running the command [^mongosh-access]
```bash
mongosh "mongodb+srv://<username>:<password>@<cluster-address>/<database-name>"
```

Replace `<username>`, `<password>`, `<cluster-address>`, and `<database-name>` with your own values.

You can find more information on how to install and use mongosh in the official MongoDB documentation [^mongosh-install] [^mongosh-access].

[^mongosh-install]: https://www.mongodb.com/docs/mongodb-shell/install/
[^mongosh-access]: https://www.mongodb.com/docs/mongodb-shell/connect/

## Get one record by searching an attribute 

**Get user by id**
```javascript
db.users.find({"_id":ObjectId("<value>")})
```

**Get user by telephone**
```javascript
db.users.find({telephone:"<telephone>"})
```

## Get all record on constraint

**Get all restaurants**
```javascript
db.restaurants.find()
```

**Get all users whose telephone contains 210**
```javascript
db.users.find({ telephone: { $regex: '210' }})
```

**Get the n latest**
```javascript
db.purchases.find({}).sort({_id:-1}).limit(1)
```

**Get the list of _id for all articles**
```javascript
db.articles.find({}, { _id: 1 }).toArray().map((doc) => doc._id)
```

**Get users whose prenoms contain `abc`, case insensitive**
```javascript
db.users.find({ prenoms: { $regex: "abc", $options: 'i' } });
```

## Find with cross tables constraints

**Get all restaurants without an owner**
```javascript
db.restaurants.aggregate([
  { $lookup: { from: "users", localField: "own_by", foreignField:"_id", as: "owner" } },
  { $match: { owner: { $size: 0 } /* Filter where "owner" array is empty, meaning no matching user found*/ } }
]);
```

**Get all users whose _id is present in the restaurants table and are of type "RESTAU"**
```javascript
db.users.aggregate([
  { $lookup: { from: "restaurants", localField: "_id", foreignField: "own_by", as: "restau" }},
  { $match: { type: "RESTAU" }}
]);
```

**Get users with at least one restaurant and apply a modification**
```javascript
db.users.aggregate([
  { $lookup: { from: "restaurants", localField: "_id", foreignField: "own_by", as: "restaurants" } },
  { $match: { restaurants: { $exists: true, $not: { $size: 0 } } /* Users with at least one restaurant*/ } },
  { $set: { key: "value" } }
]);
```

## Update rows

**Update the user with a specific _id**
```javascript
db.users.updateOne({"_id":ObjectId("<value>")}, {$set : {"nom":"new name"}})
```

**Update the user with a specific telephone**
```javascript
db.users.updateOne({telephone:"<value>"}, {$set: {"notif_data.notif_token":"<value>"}})
```

**Update the password of all users with type="RESTAU"**
```javascript
db.users.updateMany({"type":"RESTAU"}, {"$set":{"password":"<value>"}})
```

## Other Commands

**See all collections**
```javascript
show tables
```

**See all databases**
```javascript
show dbs
```

**Switch to the database "BiWag"**
```javascript
use BiWag
```

**See help**
```javascript
help
```

## Conclusion
Note: This document contains example commands and use cases for the `mongosh` shell in MongoDB. Always be cautious when performing any updates or deletions on your database and ensure you have proper backups and permissions.
