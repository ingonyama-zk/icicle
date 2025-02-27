
echo -e "\033[33mSyncing dependencies over Android Debug Bridge\033[0m"

# ${adb} push -p ./build /data/local/tmp/
adbsync  -q --exclude './icicle/_deps/**' push ./build /data/local/tmp


echo -e "\033[33mRunning example over Android Debug Bridge\033[0m"

# clear logcat messages
adb logcat -c

#  redirect logcat ICICLE message to ADB shell

# ${adb} shell logcat -v time  "ICICLE:*" "*:S"  & 

# run example

# adb shell logcat -v time  "ICICLE:*" "*:S"  &  adb shell "LD_LIBRARY_PATH=/data/local/tmp/build/icicle:/data/local/tmp/build/icicle/backend/vulkan ICICLE_BACKEND_INSTALL_DIR=/data/local/tmp/build/icicle/backend /data/local/tmp/build/example/example VULKAN"
adb shell "LD_LIBRARY_PATH=/data/local/tmp/build/icicle:/data/local/tmp/build/icicle/backend/vulkan ICICLE_BACKEND_INSTALL_DIR=/data/local/tmp/build/icicle/backend /data/local/tmp/build/example/example VULKAN"
