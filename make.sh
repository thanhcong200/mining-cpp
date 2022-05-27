# rm -rf build
if [ ! -d "/path/to/dir" ] 
then
    mkdir build
fi
cd build
cmake ..
make -j4
./NeuralNet