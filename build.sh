echo "Building project"
mkdir build 2&>dev/null
mkdir bin 2&>dev/null
cd build
cmake ..
make