# Cross-compile to Android

At this moment we only support cross-compiling Rust applications

## Cross-compile Rust CLI app to Android using Ubuntu container

### Build a container to setup the required toolchains

We assume you are in the Icicle root directory.

- on Ubuntu (amd86), run

```sh
docker build -t icicle-cross-android -f ./scripts/cross-compilation/android/Dockerfile ./scripts/cross-compilation/android/
```

- on Apple Silicon (arm64), ensure the image supports multi-platform builds:

```sh
docker buildx build --platform linux/amd64 -t icicle-cross-android -f ./scripts/cross-compilation/android/Dockerfile ./scripts/cross-compilation/android/
```

### Use the container to cross-compile

We assume you are in the root directory of your Rust project:

Edit the exact location of Icicle directory and run the container interactively:

```sh
docker run --platform linux/amd64 -it \
  -v $(pwd):/app \
  icicle-cross-android /bin/bash
```

Once inside the container:

```sh
cd /app
cargo build --target aarch64-linux-android --release
exit
```

Once you exit the container your cross-compiled application is in `./target/aarch64-linux-android`