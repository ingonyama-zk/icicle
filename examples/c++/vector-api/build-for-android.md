# How to run this on Android

The cross-compile toolchain is set up in a docker container. 

* Run ```./android-build-toolchain.sh``` once to build the container.
* Run ```./android-cross-compile.sh ``` to create the android executable 
    - this drops you into a shell in the container for now
    - ../../.icicle is mounted at /workspace/icicle
    -  run ```android-build.sh```from within the container
* Run ```./android-run.sh``` to deploy via adb.  

