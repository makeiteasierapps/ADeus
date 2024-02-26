#!/bin/bash
<<<<<<< HEAD

# Create a build directory
mkdir -p build
cd build

# Generate build files using CMake
cmake ..

# Build the project using make
make

cp -f main ..
=======
sudo g++ -std=c++17 main.cpp -g -o main -lasound -lcurl -lpthread -lFLAC
>>>>>>> bfef88f (Begin refactor)
