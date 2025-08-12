---
date: 2024-02-10
authors: [hermann-web]
comments: true
description: >
  Dive into Python logging with examples using both built-in logging and the Loguru library.
categories:
  - programming
  - python
  - logging
  - tools-comparison
  - deployment
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "Logging for Deployment in Python: A Practical Guide to Effective Debugging and Monitoring"
---

# Logging for Deployment in Python: A Practical Guide to Effective Debugging and Monitoring

## Introduction

__Are you still using print() statements for debugging in Python? Upgrade your logging game with Python's built-in logging module or the Loguru library!__

If you're tired of scattered print statements cluttering your codebase, it's time to embrace the power of logging in Python. Whether you're a beginner or an experienced developer, mastering logging techniques is essential for effective debugging, monitoring, and troubleshooting of your Python applications.

<div align="center"><div class="simple-img-container">
  <a  title="Credit: Samyak Tamrakar" href="https://github.com/srtamrakar/python-logger/blob/master/README.md"><img alt="logging-demo" src="https://raw.githubusercontent.com/srtamrakar/python-logger/master/docs/demo-logs.png"></a>
</div></div>

Logging in Python allows you to set different levels of logging, such as DEBUG, INFO, WARNING, ERROR, and CRITICAL. With these levels, you can control the verbosity of log messages and focus on the information relevant to your current task.

<!-- more -->

In this comprehensive guide, we'll explore both the built-in logging module and the [Loguru library](https://loguru.readthedocs.io), offering practical examples and best practices for seamless integration into your projects. Say goodbye to ad-hoc debugging and hello to structured and manageable logs!

## Advantages of Logging in Python

Logging in Python offers several advantages over ad-hoc debugging using print statements. Here are some key benefits:

### Structured Output

Unlike print statements, which often result in unstructured output scattered throughout the codebase, logging allows developers to generate structured logs with predefined formats. This structured output makes it easier to parse and analyze log data, leading to improved debugging and troubleshooting.

### Granular Control

With logging, developers can control the verbosity of log messages by setting different logging levels (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). This granular control enables developers to filter log messages based on their severity, allowing them to focus on relevant information and ignore less critical messages.

### Modularization

Logging encourages modularization of code by promoting the separation of concerns. By logging messages within specific modules or components, developers can track the flow of execution and identify potential issues more effectively. Additionally, modular logging facilitates collaboration among team members by providing insight into each component's behavior.

### Runtime Configuration

Logging in Python allows for runtime configuration, meaning developers can adjust logging settings without modifying the source code. This flexibility is particularly useful in scenarios where different logging configurations are required for development, testing, and production environments.

### Performance Monitoring

Logging is essential for performance monitoring and profiling of Python applications. By logging performance-related metrics such as execution times, memory usage, and resource consumption, developers can identify bottlenecks and optimize the performance of their applications.

### Error Handling

Effective error handling is crucial for robust Python applications. Logging provides a centralized mechanism for capturing and reporting errors, allowing developers to track the occurrence of exceptions and trace their origins. This helps in identifying and addressing potential issues before they impact the application's functionality.

### Integration with Monitoring Tools

Logging seamlessly integrates with monitoring and alerting tools, enabling developers to monitor the health and performance of their applications in real-time. By integrating logging with tools like Prometheus, Grafana, or ELK Stack, developers can gain valuable insights into their application's behavior and take proactive measures to maintain its reliability.

In the following sections, we'll delve deeper into the features and usage of both the built-in logging module and the Loguru library, showcasing practical examples and best practices for effective logging in Python.

## Built-in Logging

### Basic Setup

!!! example "Basic Setup with Built-in Logging"

    ```python
    import logging

    # Creating a logger
    logger = logging.getLogger()

    # Setting up a namespace (e.g., grpc)
    logging.getLogger("grpc")
    ```

### Using Handlers

Handlers in logging are responsible for taking the log records (created by loggers) and outputting them to the desired destination, such as the console, files, or network sockets. Using handlers allows developers to control where log messages are sent and how they are formatted. For example, a console handler might be used for debugging during development, while a file handler could be used to store logs for later analysis.

!!! example "Using Handlers with Built-in Logging"

    ```python
    # Adding a console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s: %(levelname)-8s %(message)s")
    console_handler.setFormatter(formatter)

    # Clearing any existing handlers and adding the console handler
    logger.handlers = []
    logger.addHandler(console_handler)

    # Setting the logging level
    logger.setLevel(logging.WARNING)  # Adjust the level as needed
    ```

### Logging Levels

Logging levels provide a way to categorize log messages based on their severity. By setting the logging level, developers can control which messages are emitted by the logger. This is essential for managing the volume of log output and focusing on relevant information. For instance, during development, setting the level to DEBUG allows developers to see detailed debugging information, while in production, it might be set to WARNING or higher to only capture critical issues.

!!! example "Logging Levels with Built-in Logging"

    ```python
    logger.debug("Debug message: Detailed diagnostic output")
    logger.info("Info message: General system information")
    logger.warning("Warning message: Something to take note of but not critical")
    logger.error("Error message: A major problem that needs attention")
    logger.critical("Critical message: A severe error indicating a major failure in the application")
    ```

!!! example "logging example using formatted string"

    ```python
    category = "module"
    name = "example_function"
    parameters = {"arg1": "value1", "arg2": "value2"}
    logger.info("%s %s: %r", category, name, parameters)
    ```

### Suppressing Logging

Sometimes during testing or specific scenarios, it's beneficial to suppress logging within certain namespaces or contexts to keep the output clean and focused on relevant information. This is especially useful when running automated tests or building processes where excessive logging could clutter the output or interfere with the test results.

To achieve this, use the `suppress_logging` context manager, which temporarily disables logging

 within the specified namespace.

!!! example "Suppressing Logging with Built-in Logging"

    ```python
    from contextlib import contextmanager

    @contextmanager
    def suppress_logging(namespace):
        """
        Suppress logging within a specific namespace to keep output clean during testing or build processes.
        """
        logger = logging.getLogger(namespace)
        old_value = logger.disabled
        logger.disabled = True
        try:
            yield
        finally:
            logger.disabled = old_value

    # Usage example:
    with suppress_logging("your_namespace"):
        # Your code using logging goes here
        pass

    ```

## Loguru

Loguru is a powerful logging library for Python that simplifies logging configuration and provides advanced features such as automatic inclusion of file, function, and line information in log messages. It offers a more intuitive and flexible logging experience compared to the built-in logging module, making it a popular choice among developers.

![](https://raw.githubusercontent.com/Delgan/loguru/master/docs/_static/img/demo.gif)

### Installation

```bash
pip install loguru
```

### Usage

!!! example "Usage with Loguru"

    ```python
    from loguru import logger

    # Logging messages with Loguru
    logger.debug("A debug message")
    logger.info("An info message")
    logger.warning("A warning message")
    logger.error("An error message")
    logger.critical("A critical message")

    # Loguru output automatically includes by default module name, function name as well as line information
    ```

## Conclusion

Logging plays a crucial role in Python development by providing a structured and manageable approach to debugging, monitoring, and troubleshooting applications. Whether using the built-in logging module or the Loguru library, developers can leverage logging to improve code quality, streamline development workflows, and enhance application reliability.

By mastering logging techniques and best practices outlined in this guide, developers can effectively manage log output, control verbosity, modularize code, monitor performance, handle errors, and integrate with monitoring tools. With structured and informative logs, Python developers can gain valuable insights into their applications, leading to more efficient development cycles and improved overall software quality.

## Related Posts

- [Cheat on Python Package Managers](../../../posts/programming-languages/python/package-managers-in-python.md)
