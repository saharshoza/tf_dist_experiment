#! /bin/bash

ERROR="./mnist_seq_error.py"
TIME="./mnist_seq_time.py"

# TODO: Loop over CPU and Memory

# Create container and get container ID
# TODO: Change CPU and Memory here per iteration
ID=`sudo docker run -d -p 8888:8888 tensorflow/tensorflow:1.7.0`
echo $ID

# Copy files to container
sudo docker cp ./mnist_seq_time.py $ID:/notebooks/
sudo docker cp ./mnist_seq_error.py $ID:/notebooks/
sudo docker cp ./utils.py $ID:/notebooks/

# Run files
# TODO: Redirect STDERR
sudo docker exec -it $ID /bin/bash -c "pwd; ls; python mnist_seq_error.py > error.out; python mnist_seq_time.py > time.out; exit"

# Copy back to host
sudo docker cp $ID:/notebooks/time.out .
sudo docker cp $ID:/notebooks/error.out .

# Kill container
sudo docker rm -f $ID
