---
date: 2023-10-02
authors: [hermann-web]
comments: true
description: >
  This blog shows interesting stuff to know
  It is flavored by mkdocs
categories:
  - Blog
  - devops
  - web
  - beginners
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: A roadmap for web developper
---


# A Roadmap for Web Development: Lessons and Stories

## Chapter 1: HTML and CSS - Crafting Digital Experiences

In the vast landscape of web development, the journey often starts with understanding the language of the web: HTML and CSS. It's more than just syntax and techniques; it's the brush and canvas, where we paint the interfaces of the digital world. I embarked on this journey through an inspiring course that not only taught me the technicalities but also guided me to build my first project, allowing me to witness the magic of transforming code into a visual experience. [Link to the course](https://openclassrooms.com/fr/courses/1603881-apprenez-a-creer-votre-site-web-avec-html5-et-css3)

<!-- more -->

## Chapter 2: JavaScript - Embracing Dynamic Interactions

JavaScript, the language that breathes life into web pages, offers endless possibilities. You can choose to skip it and learn the language in the next course (course 3), as we'll be talking about variables, loops and functions here. But it is always good to learn the basics of a programming language before using any of his frameworks. [Link to the course](https://openclassrooms.com/en/courses/6175841-apprenez-a-programmer-avec-javascript)

## Chapter 3: JavaScript for Web - Making Web Pages Dynamic

The world of web development thrives on dynamic experiences.
This course is the profect continuation in the web dev journey after html, css.
On coursera, there is a course that introduces the javascript tools we use to manipulate objects on a web page to make it less static.

Towards the end, there's a quiz where you do a project using Node JS (a framework used to run javascript code) but I notice they've adopted a new, simpler format. [Link to the course](https://openclassrooms.com/fr/courses/5543061-ecrivez-du-javascript-pour-le-web/5577726-optimisez-votre-code)

## The Challenge: Testing Time - Cloning a Web Page

## Chapter 4: Python: Unsupervised ML Algorithms

My journey expanded beyond web development into the realms of Python. This detour into the world of unsupervised machine learning algorithms was not entirely new. It offered a different perspective, unlocking doors to exploring and analyzing data with a sense of wonder. [Link to the course](https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises/4379571-partitionnez-vos-donnees-avec-dbscan)

## Chapter 5: PHP for Backend - Connecting Frontend to Databases

The bridge between frontend and databases fascinated me. PHP became my vessel in understanding the interaction between a website and a database. While focusing on MySQL, I explored the installations and syntax, essential for crafting PHP code. [Link to reference](https://www.w3schools.com/php/)

I must admit, i learning out of pure curiosity, after my first internship in cloud computing. I wanted to know more are a crucial component of the 3-tiers architecture, which is the database and two other reseons as important

- i know php is one of the most use tools for backend development
- i know php integrate very well with html, css
- i've learned html, css, js before hand

My only experiences with php, for at least years, after learning was about helping every time a student come to be with a problem to fix or a functionnality to add.
But after a while, i got into a project where we decided with the team, to use laravel. Laravel is based on php. I got onto it very quick because, even if i have 0 real project experience in php, i've coded in php to help many students. All the tools needed was already on my computer and i've added the first feature assigned to me within a week while the others, took at least 3 weeks to undertand the framework and add the feature assigned to them.

## Chapter 6: Django for Backend in Python - Navigating the Web Development Framework

It's a Python framework for web development. I took part in a course on Coursera to learn about Django architecture (how to create a project, how files are organized, what each file is used for).

I learned a lot more about the framework during my internship at Safran than during the apprenticeship.
With the bugs to slove and features to be added, I needed to read stackoverflow discussions, try and fail, visit the documentation very often (very rich by the way, cudo to django project team).
So the real learning began amidst challenges and practical applications.

## Second internship: Web development

That was the internship i needed to be sure i wasn't just a part time web development learner. My task was to develop a web app that will enable members of a department (i wont say more) to search and access useful document through a simple search engine adpated to their need.
The project started as a django project. Before coding, i did design the web app with figma. After, the web app validated, i started the core features.

!!!
  If i have learn one important thing during the first month, it is to start with the login components if login is a required part of the process.

I've developed iteratively the web app. Even though i was told to not waste time on frontend, i could just stand ugly pages. So i was adding css/bootstrap codes of my own.
In the second month, i have mainly worked on three backend features

- one that give informations about a document. In that part of the app, users can also appli some modifications. By modifications, i'm talking about the classic ones you can make on a word document (change colors of some words, letters, surligne, bold, italic, ...).
I've use a js tool as i recall.
- the frontend of the search engine: As simple of the google search interface. I've removed my styles on this.
- By the end of the month, i started working on the search engine. Well, the web dev project quickly become a Natural language processing project too. Without going into the details, i was doing NLP with javascript then switched to python to use sklearn methods.

## Chapter 7: Bootstrap - Streamlining CSS Development

Bootstrap is a framework that makes it easy to add CSS styles to a web page. But, skip to this course if you're working on a project where you need it. You'll learn faster. Same advice as for the next framework: Jquerry. For my part, I needed it to speed up the frontend part of my internship, so I went to the w3schools site. [Link to a course](https://www.w3schools.com/bootstrap5/)
I figured long after, tailwing, as a rigid alternative to bootstrap

## Chapter 8: jQuery - Simplifying JavaScript

While JavaScript is the heart of web interactions, jQuery offered a streamlined path. Mastering it could be an asset in certain scenarios, enhancing the development process.
JQuerry is a framework that makes it easy to write javascript code for the web. Some developers don't know Jquerry as well as they know javascript, but in that case it's not always easy to work in a team that doesn't use Jquerry. [Link to a course](https://www.w3schools.com/jquery/)

## Chapter 9: Ajax - Seamlessly Connecting Frontend and Backend

Ajax, the invisible thread connecting frontend and backend without the need for page reloads. This technology became essential in my journey, especially in tandem with the Django framework, enabling seamless exchanges and a more interactive web experience. [Link to a course](https://www.w3schools.com/xml/ajax_intro.asp)

## Chapter 10: ReactJS - Shaping Dynamic User Experiences

ReactJS is a NodeJS framework used a lot by frontend developers.
A lot of developers learn reactJS, right after html, css and js, which puts them on the right track for front-end development.
React Js became a vital part of my skill set during a significant gap year. It was very much in demand and used on the market in 2022. But the real reason I decided to learn ReactJS was for an internship in 2021-2022. My first assignment was to develop a web interface. I had a choice between reactJS and AngularJS. After discussions with the lead dev, we chose ReactJS. Long story short, we wanted the user to have a one-page experience on the application.

## Chapter 11: NodeJS - Unleashing JavaScript on the Server

Javascript is well known for the front-end, but was limited to it until the development of NodeJS.
NodeJS is a tool for executing JS scripts on the server side, opening doors to multifaceted web development.

## Chapter 12: ElectronJS - Exploring Beyond the Browser

## Chapter 13: Laravel - Embracing a New Horizon

The journey continues with Laravel, a framework that beckons with new possibilities and a new chapter to explore.

Each chapter in this roadmap of web development is more than a course; it's a story, an experience, a transformation that shaped me into the software engineer I am today. The roadmap evolves, the journey continues, and the quest for learning never ceases.
