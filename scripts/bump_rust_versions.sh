#!/bin/bash

new_version=$1

if [ -z "$new_version" ]; then
  echo "Usage: ./bump_rust_versions.sh <new_version>"
  exit 1
fi

cd wrappers/rust

# Update the version in each member crate's Cargo.toml
for crate in $(cat Cargo.toml | grep '"[a-z].*"' | tr -d '[" ],'); do
    sed -i "/^\[package\]/,/^$/ s/^version = \".*\"/version = \"$new_version\"/" $crate/Cargo.toml
done