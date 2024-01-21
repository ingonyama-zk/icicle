# ZKContainer

We recommend using [ZKContainer](https://ingonyama.com/blog/Immanuel-ZKDC), where we have already preinstalled all the required dependencies, to run Icicle examples. 
To use our containers you will need [Docker](https://www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).

In each example directory, ZKContainer files are located in a subdirectory `.devcontainer`. 

- File `Dockerfile` specifies how to build an image of a ZKContainer. 
- File `devcontainer.json` enables running ZKContainer from Visual Studio Code.

## Running ZKContainer from shell

```sh
docker build -t icicle-example-poseidon -f .devcontainer/Dockerfile .
```

To run the example interactively, start the container

```sh
docker run -it --rm --gpus all -v .:/icicle-example icicle-example-poseidon
```

Inside the container, run the commands for building the library for whichever [build system](../README.md#build-systems) you choose to use. 
