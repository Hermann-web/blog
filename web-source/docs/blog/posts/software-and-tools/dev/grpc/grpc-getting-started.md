---
date: 2023-12-31
authors: [hermann-web]
comments: true
description: |
  An introductory guide to working with gRPC in Python, covering installation, testing examples, and expanding functionality through protocol buffers.
categories:
  - devops
  - microservices
  - gprc
  - python
  - networking
title: "A Beginner's Guide to gRPC with Python"
# icon: https://grpc.io/img/landing-2.svg
---

# A Beginner's Guide to gRPC with Python

## Introduction to gRPC

__Have you heard of gRPC, high-performance, open-source framework that allows developers to build distributed systems and microservices ?__

[gPRC](https://github.com/grpc/grpc) uses protocol buffers as its interface definition language and provides features such as bi-directional streaming and flow control.

<p align="center">
  <img src="https://grpc.io/img/landing-2.svg" alt="Description of the image" />
</p>

<!-- more -->

Credit: [gRPC](https://grpc.io/docs/guides/)

In this blog post, we will explore how to get started with gRPC in Python using the official gRPC Python library. We will walk through a simple working example that demonstrates how to:

1. Define a service in a `.proto` file
2. Generate server and client code using the protocol buffer compiler
3. Use the Python gRPC API to write a simple client and server for your service

### Advantages of gRPC

gRPC offers several advantages, making it a versatile and efficient choice for building distributed systems and microservices:

- **Language Independence**: gRPC supports multiple languages seamlessly, allowing developers to build distributed systems using their preferred programming language.
- **Open Source & Multilingual Support**: Being open source, gRPC enjoys support across various programming languages, making it a widely adopted solution for building distributed systems.
- **Boilerplate Elimination**: gRPC generates code, reducing the need for boilerplate code and simplifying the development process.
- **Efficient Data Encoding**: gRPC utilizes buffers instead of JSON for data encoding, resulting in lighter data transmission.

## Getting Started with gRPC in Python

### Quick Setup

Follow the steps below to set up a Python environment for gRPC [^grpc-python-quickstart]:

1. **Quick Setup:**
   
   ```bash
   cd path/to/my/folder
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. **Install gRPC and gRPC Tools:**
   
   ```bash
   python -m pip install grpcio
   python -m pip install grpcio-tools
   ```

   - **gRPC Tools Include:**
     - `protoc`, the buffer compiler
     - A plugin to generate client and server-side code from `.proto` files.

### Testing an Example

Clone the gRPC repository to access a sample project:

```bash
git clone -b v1.60.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc
```

Run the server in one terminal:

```bash
cd grpc/examples/python/helloworld
python greeter_server.py
```

The output
??? output
    ```plaintext
    Server started, listening on 50051
    ```

Run the client in another:

```bash
cd grpc/examples/python/helloworld
python greeter_client.py
```

??? output
    ```plaintext
    Will try to greet world ...
    Greeter client received: Hello, you!
    ```

Congratulations! You've run your first gRPC application!

### What the code does

The provided code includes a Protocol Buffers (protobuf) definition, a server-side implementation in Python, and a client-side implementation in Python. The protobuf definition defines a `Greeter` service with three RPC methods: `SayHello`, and `SayHelloStreamReply`. The server-side implementation defines the behavior of the `SayHello` method, while the client-side implementation makes use of these methods to communicate with the server.

The `helloworld.proto` file defines the `Greeter` service with three RPC methods. The `greeter_server.py` file implements the server for the `Greeter` service, and the `greeter_client.py` file implements the client to communicate with the server. The `python -m grpc_tools.protoc` command is used to compile the `.proto` file and generate the necessary Python code for the server and client.


### Adding an Extra Method on the Server

- Modify [`../../protos/helloworld.proto`](grpc/examples/protos/helloworld.proto) and the files `greeter_server.py` and `greeter_client.py` in the `examples/python/helloworld` folder.

=== ":octicons-file-code-16: `examples/protos/helloworld.proto`"

    ```proto
    ...
    service Greeter {
      // Sends a greeting
      rpc SayHello (HelloRequest) returns (HelloReply) {}
      
      // Sends another greeting
      rpc SayHelloAgain (HelloRequest) returns (HelloReply) {}
      rpc SayHelloStreamReply (HelloRequest) returns (stream HelloReply) {}
      ...
    }
    ...
    ```

=== ":octicons-file-code-16: `greeter_server.py`"

    ```python
    ...
    class Greeter(helloworld_pb2_grpc.GreeterServicer):

        def SayHello(self, request, context):
            return helloworld_pb2.HelloReply(message=f"Hello, {request.name}!")

        def SayHelloAgain(self, request, context):
            return helloworld_pb2.HelloReply(message=f"Hello again, {request.name}!")
    ...
    ```


=== ":octicons-file-code-16: `greeter_client.py`"

    ```python
    ...
    def run():
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = helloworld_pb2_grpc.GreeterStub(channel)
            response = stub.SayHello(helloworld_pb2.HelloRequest(name='you'))
            print("Greeter client received: " + response.message)
            response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name='you'))
            print("Greeter client received: " + response.message)
    ...
    ```

- Compile the `.proto` file and generate the necessary Python code for the server and client
```bash
python -m grpc_tools.protoc -I../../protos --python_out=. --pyi_out=. --grpc_python_out=. ../../protos/helloworld.proto
```

- Run the client and server again:

```bash
python greeter_server.py
python greeter_client.py
```

### What just happened 

Well, we have added another RPC method, called here `SayHelloAgain`. The implementation includes:

- The protobuf definition in the `Greeter` service in the `greeter_server.py`
- The server-side implementation in `greeter_server.py`

So, when running the server then the client, we should receive two responses

The server output should remain the same
```plaintext
Server started, listening on 50051
```

But the client will receive two responses from the server.
```plaintext
Will try to greet world ...
Greeter client received: Hello, you!
Greeter client received: Hello again, you!
```


The `python -m grpc_tools.protoc` command is used to compile the `.proto` file and generate the necessary Python code for the server and client. This command takes the following arguments:
- `-I../../protos`: Specifies the directory containing the `.proto` file.
- `--python_out=.`: Specifies the output directory for the generated Python code.
- `--grpc_python_out=.`: Specifies the output directory for the generated gRPC Python code.

This command generates the `helloworld_pb2.py` file, which contains the generated request and response classes, and the `helloworld_pb2_grpc.py` file, which contains the generated server and client stubs.

The `python -m grpc_tools.protoc` command is the recommended way to generate Python code from a `.proto` file for use with gRPC.

For more information, you can refer to the gRPC Python documentation and the Protocol Buffer Basics: Python tutorial.

If you need to compile `.proto` files for other programming languages, the process may differ, and you can refer to the respective language's gRPC documentation for guidance.


## Further Reading:

- [Introduction to gRPC](https://grpc.io/docs/what-is-grpc/introduction/)
- [gRPC Core Concepts](https://grpc.io/docs/what-is-grpc/core-concepts/)
- [Explore the Python API Reference](https://grpc.io/docs/languages/python/api) to discover functions and classes.

For more detailed instructions, refer to the gRPC Python Quickstart [^grpc-python-quickstart].

[^grpc-python-quickstart]: https://grpc.io/docs/languages/python/quickstart/
[^1]: https://stackoverflow.com/questions/62649353/difference-between-protoc-and-python-m-grpc-tools-protoc
[^3]: https://grpc.io/docs/languages/python/generated-code/
[^4]: https://stackoverflow.com/questions/57909401/what-are-the-command-line-arguments-passed-to-grpc-tools-protoc
[^0]: https://grpc.io/docs/languages/python/basics/
[^5]: https://www.velotio.com/engineering-blog/grpc-implementation-using-python
[^6]: https://realpython.com/python-microservices-grpc/