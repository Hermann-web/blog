---
date: 2023-10-30
authors: [hermann-web]
comments: true
description: >
  Explore the fundamental concepts and best practices of JavaScript in this beginner-friendly guide, designed for those new to the language or seeking to refresh their knowledge. Covering topics such as data types, control structures, functions, error handling, and more.
categories:
  - programming
  - javascript
  - web
  - fundamentals
title: "JavaScript Fundamentals: A Beginner's Guide to Essential Concepts and Best Practices"
---

# JavaScript Fundamentals: A Beginner's Guide to Essential Concepts and Best Practices

## Introduction

__Are you new to JavaScript or looking to refresh your understanding of its fundamental concepts?__

Dive into this beginner-friendly guide to explore the core principles and best practices in JavaScript!

<div class="float-img-container float-img-right">
    <a title="Lionel Rowe, CC0, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:JavaScript_code.png"><img width="512" alt="JavaScript code" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/JavaScript_code.png/512px-JavaScript_code.png"></a>
</div>

JavaScript (ES6), short for ECMAScript 6, is a major version of the JavaScript programming language. It introduces new features, syntax enhancements, and advanced programming concepts. This documentation is tailored for beginners and individuals seeking to recall the fundamentals of JavaScript. Let's embark on this learning journey together!

### Overview

In this comprehensive guide, you will learn:

- Essential JavaScript data types, control structures, and functions
- Error handling techniques and best practices
- The use of classes, objects, arrays, and maps in JavaScript
- Comparison operators and logical operations
- Control structures including if/else, switch/case, for loops, and while loops
- Best practices for variable declaration and scope
- Tips for writing clean and maintainable JavaScript code

Ready to elevate your JavaScript skills? Let's get started!

<!-- more -->

## Variables and Constants

### Declaration and Initialization

- Use `let` to declare variables whose value can change.
- Use `const` to declare constants whose value cannot change.

Example:

```javascript
let numberOfEpisodes = 9;
const pi = 3.14;
```

## Data Types

- Data types include `number`, `string`, `boolean`, `object`, `undefined`, `null`, etc.
- Use `typeof` to check the type of a variable.

### Numbers

In JavaScript, integers are represented as the `number` data type. You can declare a variable and assign it an integer value as follows:

```javascript
let a = 15;
console.log(typeof a);  // Displays "number"
```

### Booleans

Booleans represent true or false values in JavaScript. You can declare a variable and assign it a boolean value like this:

```javascript
let userIsSignedIn = true;
console.log(typeof userIsSignedIn); // Displays "boolean"
```

### Strings

Strings are sequences of characters enclosed in single or double quotes. You can concatenate two strings using the `+` operator. Here's an example:

```javascript
let nom = 'Ag';
let prenom = 'He';
console.log(typeof nom); // Displays "string"
console.log(nom + prenom); // Displays "AgHe"
```

## Outputs

- Use `console.log()` to display messages in the console.
- Use `alert()` to display messages in a dialog box.

Example:

```javascript
console.log(5);
let yyy = "me";
console.log("retdy" + " " + yyy);
alert('some things'); // Display a message in a dialog box
```

## Objects

JavaScript objects behave similarly to dictionaries in Python.

### Syntax

Objects in JavaScript are declared using curly braces `{}` and consist of key-value pairs. Keys are always strings. Each value can be an instance of any other type.

```javascript
let mybook = { 
    title: 'Allah is not obliged',
    author: 'Un Mec',
    numberOfPages: 200,
    isAvailable: true
};
```

### Accessing Data

You can access data in an object using dot notation (`.`) or square bracket notation (`[]`). For example:

=== ":octicons-file-code-16: using dot notation (`.`)"

    ```javascript
    let titre = mybook.title;
    let auteur = mybook.author;
    let isdisponible = mybook.isAvailable;
    ```
=== ":octicons-file-code-16: using bracket notation (`[]`)"

    ```javascript
    let titre = mybook["title"];
    let auteur = mybook["author"];
    let isdisponible = mybook["isAvailable"];
    ```

Note that the keys are case-sensitive. So `mybook.title` is not the same as `mybook.Title`.

## Classes

In JavaScript, classes allow you to define blueprints for creating objects with properties and methods.

### Class Definition

Classes are defined using the `class` keyword followed by the class name.

```javascript
class Book {
    constructor(title, author, numberOfPages) {
        this.title = title;
        this.author = author;
        this.numberOfPages = numberOfPages;
    }
}
```

### Creating Instances

You can create new instances of a class using the `new` keyword followed by the class name and passing the required parameters to the constructor.

```javascript
let aBook = new Book("le moi intérieur", "Hermann", 250);
```

### Methods

Methods can be defined within the class using normal function syntax. Here's an example:

??? example "method example"

    ```javascript
    class BankAccount {
        constructor(owner, balance) {
            this.owner = owner;
            this.balance = balance;
        }
        
        showBalance() {
            console.log("Solde: " + this.balance + " EUR");
        }
        
        deposit(amount) {
            console.log("Dépôt de " + amount + " EUR");
            this.balance += amount;
            this.showBalance();
        }
        
        withdraw(amount) {
            if (amount > this.balance) {
                console.log("Retrait refusé !");
            } else {
                console.log("Retrait de " + amount + " EUR");
                this.balance -= amount;
                this.showBalance();
            }
        }
    }
    ```

### Instantiation and Method Invocation

You can create an instance of a class and then call its methods as shown below:

```javascript
const newAccount = new BankAccount("Will Alexander", 500);
newAccount.showBalance(); // Prints "Solde: 500 EUR" to the console
```

This code creates a new bank account with an initial balance of 500 EUR and then displays its balance using the `showBalance` method.

## Arrays and Maps

In JavaScript, arrays are used to store collections of elements. They are zero-indexed, meaning the index of the first element is 0.

### Array Definition

Arrays can be defined without initialization, or directly initialized with elements.

```javascript
// Definition without initialization
let myList;

// Definition with direct initialization
let guests = [];
let invitedPeoples = ["Sarah", "Jean-Pierre", "Claude"];
```

### Accessing Elements

You can access elements in an array using square brackets and the index of the element. Remember that array indexing starts from 0.

```javascript
let guest1 = invitedPeoples[0]; // Accesses the first element
let guest3 = invitedPeoples[2]; // Accesses the third element
```

### Array Length

You can determine the length of an array using the `length` property.

```javascript
console.log(invitedPeoples.length); // Prints the length of the array
```

### Array Operations

Arrays support various operations like adding elements, removing elements, and inserting elements.

- To add an element to the end of an array, you can use the `push` method.
- To remove the last element from an array, you can use the `pop` method.
- To add an element to the beginning of an array, you can use the `unshift` method.

### Maps

Maps in JavaScript are similar to arrays but are unordered collections. They do not allow duplicates, and you can check if an element exists in a map.

These are some basic operations you can perform with arrays in JavaScript.

## Comparison Operators

Comparison operators in JavaScript allow you to compare values and determine the relationship between them. These operators are commonly used in conditional statements like `if` and loops.

### Common Comparison Operators

- `<`    Less than
- `<=`   Less than or equal to
- `>`    Greater than
- `>=`   Greater than or equal to
- `==`   Equal to (checks value only)
- `!=`   Not equal to (checks value only)
- `===`  Equal to (checks value and type)
- `!==`  Not equal to (checks value and type)

### Logical Operators

Logical operators allow you to combine multiple conditions in a single statement.

- `&&`   Logical AND: Returns `true` if both conditions are true
- `||`   Logical OR: Returns `true` if at least one condition is true
- `!`    Logical NOT: Negates the result, returns `true` if the condition is false

!!! example "comparison evaluation"

    ```javascript
    let a = 5;
    let b = 10;

    if (a < b) {
        console.log("a is less than b");
    } else {
        console.log("a is greater than or equal to b");
    }
    ```

    In this example, the condition `a < b` evaluates to `true`, so the message "a is less than b" will be logged to the console.

## Control Structures

## If/else

In JavaScript, `if`, `else if`, and `else` statements are used to execute blocks of code based on conditions.

### Basic Structure

The basic structure of an `if` statement is as follows:

```javascript
if (condition) {
    // Code to execute if the condition is true
} else {
    // Code to execute if the condition is false
}
```

??? example "example of using an `if` statement"

    ```javascript
    let userLoggedIn = true;

    if (userLoggedIn) {
        console.log("User logged in!");
    } else {
        console.log("Alert, intruder!");
    }
    ```

    In this example, if the `userLoggedIn` variable is `true`, the message "User logged in!" will be logged to the console; otherwise, "Alert, intruder!" will be logged.

??? info "multiple conditions with if / else if / else"

    You can also use `else if` statements to test multiple conditions. Here's an example:

    ```javascript
    if (numberOfGuests == numberOfSeats) {
        // All seats are occupied
    } else if (numberOfGuests < numberOfSeats) {
        // Allow more guests
    } else {
        // Do not allow new guests
    }
    ```

    In this example, if the number of guests equals the number of seats, the first block of code will execute. If the number of guests is less than the number of seats, the second block of code will execute. Otherwise, the third block of code will execute.

### Conditions Evaluated as True or False

=== ":octicons-file-code-16: Conditions Evaluated as True"

    Conditions that can be evaluated as true in an `if` statement include:

    - Numbers that are not zero
    - Strings that are not empty
    - Boolean `true`
    - Objects (including arrays and functions)

=== ":octicons-file-code-16: Conditions Evaluated as False"

    Conditions that are evaluated as false in an `if` statement include:

    - Number `0`
    - Empty string `''`
    - Boolean `false`
    - `null`
    - `undefined`
    - `NaN`

## Switch/Case

The `switch` statement in JavaScript allows you to execute different blocks of code based on different conditions. It's particularly useful when you have a single value that you want to compare to multiple possible variants.

### Basic Structure

The basic structure of a `switch` statement looks like this:

```javascript
switch (expression) {
    case value1:
        // Code to execute if expression === value1
        break;
    case value2:
        // Code to execute if expression === value2
        break;
    default:
        // Code to execute if expression doesn't match any case
}
```

??? example "swich case example"

    ```javascript
    let guestType = "star";
    let vipStatus;

    switch (guestType) {
        case "artist":
            vipStatus = "Normal";
            break;
        case "star":
            vipStatus = "Important";
            break;
        default:
            vipStatus = "None";
    }

    console.log("VIP status:", vipStatus);
    ```

    In this example, if the `guestType` is "artist", the `vipStatus` will be set to "Normal". If the `guestType` is "star", the `vipStatus` will be set to "Important". Otherwise, the `vipStatus` will be set to "None".

### Handling Unknown Values

The `default` case is used to handle values that don't match any of the specified cases. This is useful for providing a fallback option or handling unexpected input.

??? warning "Importance of Break"

    It's important to include `break` statements after each case block to prevent fall-through behavior, where execution continues to the next case block regardless of whether the condition is met. Here's an example illustrating the importance of `break`:

    ```javascript
    let vipStatus = "";
    let guest = {
        name: "Sarah Kate",
        age: 21,
        ticket: true,
        guestType: "artist"
    };

    switch (guest.guestType) {
        case "artist":
            vipStatus = "Normal";
        case "star":
            vipStatus = "Important";
            break;
        case "presidential":
            vipStatus = "Mega-important";
            break;
        default:
            vipStatus = "None";
    }
    ```

    In this example, the `vipStatus` variable is erroneously assigned "Normal" because the `break` statement is missing after the `"artist"` case. Without the `break`, execution falls through to the `"star"` case, causing `vipStatus` to be overwritten with "Important". To avoid this, ensure that each case block ends with a `break` statement.

## For Loops

In JavaScript, `for` loops are used to iterate over elements in an array or perform a specific action a certain number of times.

### Basic Structure

The basic structure of a `for` loop is as follows:

```javascript
for (initialization; condition; increment/decrement) {
    // Code to execute for each iteration
}
```

!!! example "for loop: basic usage example"

    ```javascript
    for (let i = 0; i < numberOfPassengers; i++) {
        console.log("Passenger boarded!");
    }
    ```

### For Loop on Arrays

This section provides an overview of `for` loops in JavaScript, including examples of both `for...in` and `for...of` loops and their respective use cases

??? example "Example 1: Using `for...in` Loop"

    The `for...in` loop iterates over the enumerable properties of an object, such as the indices of an array. Here's an example:

    ```javascript
    const passengers = [
        "Will Alexander",
        "Sarah Kate",
        "Audrey Simon",
        "Tao Perkington"
    ]

    for (let i in passengers) {
        console.log("Boarding passenger: " + passengers[i]);
    }
    ```

??? example "Example 2: Using `for...of` Loop"

    The `for...of` loop is used to iterate over iterable objects, such as arrays. It provides a more concise syntax compared to the `for...in` loop. Here's an example:

    ```javascript
    const passengers = [
        "Will Alexander",
        "Sarah Kate",
        "Audrey Simon",
        "Tao Perkington"
    ]

    for (let passenger of passengers) {
        console.log("Boarding passenger: " + passenger);
    }
    ```

In both examples, each passenger's name is logged to the console, indicating that they are boarding the vehicle or entering some other context.

## While Loops

In JavaScript, `while` loops are used to execute a block of code repeatedly as long as a specified condition is true.

### Basic Structure

The basic structure of a `while` loop is as follows:

```javascript
while (condition) {
    // Code to execute as long as the condition is true
}
```

??? example "Example: Using a While Loop"

    Here's an example of using a `while` loop to repeatedly perform a task until a condition is no longer true:

    ```javascript
    let seatsLeft = 10;
    let passengersStillToBoard = 8;
    let passengersBoarded = 0;

    while (seatsLeft > 0 && passengersStillToBoard > 0) {
        passengersBoarded++; // A passenger boards
        passengersStillToBoard--; // Decrease the number of passengers still to board
        seatsLeft--; // Decrease the number of seats left
    }

    console.log(passengersBoarded); // Logs the total number of passengers boarded
    ```

    In this example, the loop continues as long as there are seats available (`seatsLeft > 0`) and passengers still to board (`passengersStillToBoard > 0`). Each iteration of the loop represents a passenger boarding the vehicle or entering some other context.

    The loop terminates when either there are no more seats available or there are no more passengers to board.

## Error Handling

In JavaScript, error handling is crucial for managing unexpected situations or errors that may occur during code execution. There are various types of errors, and JavaScript provides mechanisms like `try` and `catch` to handle them gracefully.

??? note "Types of Errors"

    1. **Syntax Errors**: These errors occur when there is a mistake in the syntax of the code, such as missing semicolons `;`, brackets `{}`, or incorrect expressions.

    2. **Logical Errors**: Logical errors happen when the code executes but produces unexpected results due to incorrect logic or reasoning in the code.

    3. **Runtime Errors**: Runtime errors occur during the execution of the program, typically caused by factors such as incorrect user input, resource unavailability, or unexpected behavior of external dependencies.

    4. **Reference Errors**: Reference errors occur when trying to access variables or functions that are not declared or out of scope.

    5. **Type Errors**: Type errors occur when an operation is performed on a value of the wrong type, such as using a method on a non-object or passing incorrect arguments to a function.

    6. **Range Errors**: Range errors occur when trying to access an invalid index of an array or perform an invalid operation within a certain numeric range.

### Exception Handling with Try/Catch

JavaScript provides the `try` and `catch` blocks for handling exceptions and managing errors effectively.

#### Basic Structure

```javascript
try {
    // Code that may cause an error
} catch (error) {
    // Handling the error
}
```

In this structure:

- The `try` block contains the code that might throw an error.
- If an error occurs within the `try` block, control is transferred to the `catch` block.
- The `catch` block is responsible for handling the error. It receives the error object as a parameter, which can be used to identify and respond to the error appropriately.

??? example "Example"

    ```javascript
    try {
        // Attempting to execute code that may throw an error
        let result = 10 / 0; // This will cause a division by zero error
        console.log(result); // This line will not execute due to the error
    } catch (error) {
        // Handling the error
        console.error("An error occurred:", error.message);
    }
    ```

    In this example, if a division by zero error occurs within the `try` block, the control will be transferred to the `catch` block. The `catch` block then handles the error by logging a descriptive message to the console.

Error handling with `try` and `catch` is an essential aspect of writing robust JavaScript code, ensuring that your applications can gracefully handle unexpected errors and provide a better user experience.

## Functions

In JavaScript, functions are blocks of reusable code that can be invoked (called) to perform a specific task. They play a crucial role in organizing and structuring code, making it easier to manage and maintain.

### Basic Syntax

Functions in JavaScript can be defined using different syntaxes, including arrow functions and traditional function declarations.

=== ":octicons-file-code-16: `Arrow Functions:`"

    Arrow functions are a concise way to write functions in JavaScript. They are commonly used for short, single-expression functions.
    
    ```javascript
    const sum = (a, b) => {
        return a + b;
    };
    ```

=== ":octicons-file-code-16: `Traditional Function Declarations:`"

    Traditional function declarations use the `function` keyword to define functions.
    
    ```javascript
    function sum(a, b) {
        return a + b;
    }
    ```

### Function Invocation

Once a function is defined, it can be invoked (called) by its name, followed by parentheses containing any arguments required by the function.

#### Example

```javascript
const result = sum(3, 4);
console.log(result); // Output: 7
```

??? example "Example 1: Function to Calculate Sum"

    ```javascript
    const sum = (number1, number2) => {
        const result = number1 + number2;
        return result;
    }

    console.log(sum(4, 7)); // Output: 11
    ```

    In this example, the `sum` function takes two parameters `number1` and `number2`, calculates their sum, and returns the result.

??? example "Example 2: Function to Calculate Average Rating"

    ```javascript
    const calculateAverageRating = (ratings) => {
        if (ratings.length === 0) {
            return 0;
        }
        
        let sum = 0;
        for (let rating of ratings) {
            sum += rating;
        }
        
        return sum / ratings.length;
    }

    const tauRatings = [5, 4, 5, 5, 1, 2];
    const colinRatings = [5, 5, 5, 4, 5];

    const tauAverage = calculateAverageRating(tauRatings);
    const colinAverage = calculateAverageRating(colinRatings);

    console.log(tauAverage); // Output: 3.6666666666666665
    console.log(colinAverage); // Output: 4.8
    ```

    In this example, the `calculateAverageRating` function calculates the average rating based on the provided array of ratings.

!!! note "Additional Notes"

    - It's common to use arrow functions (`=>`) for defining functions in modern JavaScript.
    - Constants declared with `const` are used to define functions to prevent accidental reassignment.
    - Functions can take parameters and return values, making them versatile and powerful tools for organizing code.

!!! quote "Functions are essential for structuring JavaScript code and promoting code reuse and maintainability."

### Scope of Variables

In JavaScript, the scope of a variable determines where the variable is accessible within the code. Understanding variable scope is essential for writing maintainable and bug-free code.

### Variable Declaration

When declaring variables in JavaScript, it's recommended to use `let` or `const` to define variables.

- __`let`__: Variables declared with `let` have block scope, meaning they are only accessible within the block (enclosed by `{}`) in which they are defined, as well as any nested blocks (e.g., inside an `if` statement or loop) within that block.

- __`const`__: Constants declared with `const` also have block scope and cannot be reassigned. They follow the same scoping rules as variables declared with `let`.

- __`var`__: Variables declared with `var` have function scope. This means they are accessible throughout the entire function in which they are defined, regardless of block boundaries.

??? example "Example"

    ```javascript
    {
        let localVar = 'I am a local variable';
        console.log(localVar); // Output: 'I am a local variable'
    }

    console.log(localVar); // Throws ReferenceError: localVar is not defined
    ```

    In this example, the variable `localVar` is declared using `let` inside a block. It is accessible within that block but not outside of it. Attempting to access `localVar` outside the block results in a `ReferenceError`.

??? warning "Avoid using `var` for variable declaration whenever possible"

    Using `let` and `const` for variable declaration helps prevent accidental variable hoisting and unintended side effects. It also promotes better code readability and maintenance by clearly defining the scope of variables.

    Avoid using `var` for variable declaration whenever possible, as it can lead to unexpected behavior due to its function scope and variable hoisting characteristics.

    Understanding variable scope is crucial for writing clean, predictable, and bug-free JavaScript code.

!!! abstract "Variables declared within a function are only accessible within that function, unless they are declared using the `var` keyword. Using `let` or `const` ensures that variables have block scope, making them accessible only within the block they are defined in."

!!! tip "Best Practices"

    - Use meaningful variable names.
    - Indent your code properly to make it readable.
    - Comment your code to explain its functionality.
    - Avoid ambiguous variable names.

!!! abstract "Additional Notes"

    - Remember to use semicolons to terminate statements.
    - Use `{}` to define code blocks.
    - Be mindful of the scope of variables when using `let`, `const`, and `var`.
    - Always handle exceptions to prevent unexpected behavior.
    - Utilize console methods such as `console.error()` for error messages.

??? abstract "Resources"

    Here are some useful resources to enhance your JavaScript skills and productivity:

    - [ECMAScript Compatibility Table](https://kangax.github.io/compat-table/es6/): A comprehensive table detailing the compatibility of various ECMAScript features across different JavaScript engines and environments.

    - [JS Bin](https://jsbin.com/?js,console): An online tool for quickly experimenting with and testing JavaScript code snippets. It provides a live-coding environment with a built-in console for immediate feedback.

    - [W3Schools JavaScript Tutorial](https://www.w3schools.com/js/): W3Schools offers a comprehensive and beginner-friendly JavaScript tutorial covering all fundamental concepts, syntax, and features of the language.

    - [OpenClassrooms JavaScript Course](https://openclassrooms.com/en/courses/5664271-learn-programming-with-javascript) (English) or [in French](https://openclassrooms.com/en/courses/7696886-apprenez-a-programmer-avec-javascript): OpenClassrooms provides interactive JavaScript courses suitable for beginners and intermediate learners. These courses cover topics ranging from basic syntax to advanced JavaScript programming techniques.

    These resources offer valuable insights, tutorials, and tools to help you master JavaScript programming and become a proficient developer.
