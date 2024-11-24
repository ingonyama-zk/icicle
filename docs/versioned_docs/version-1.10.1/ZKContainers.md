# ZKContainer

We found that developing ZK provers with ICICLE gives developers the ability to scale ZK provers across many machines and many GPUs. To make this possible we developed the ZKContainer.

## What is a ZKContainer?

A ZKContainer is a standardized, optimized and secure docker container that we configured with ICICLE applications in mind. A developer using our ZKContainer can deploy an ICICLE application on a single machine or on a thousand GPU machines in a data center with minimal concerns regarding compatibility.

ZKContainer has been used by Ingonyama clients to achieve scalability across large data centers.
We suggest you read our [article](https://medium.com/@ingonyama/product-announcement-zk-containers-0e2a1f2d0a2b) regarding ZKContainer to understand the benefits of using them.

![ZKContainer inside a ZK data center](/img/architecture-zkcontainer.png)
