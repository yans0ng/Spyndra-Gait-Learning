# Spyndra-Gait-Learning
This project aims at generating walking locomotion of legged robot autonomously. The training data were generated on the open-source robotic platform [Spyndra](http://www.creativemachineslab.com/spyndra.html).

## Dependencies
* python 2.7
* tensorflow
* keras

## Installation

1. First, download the repository.

   ```
   $ git clone https://github.com/roboticistYan/Spyndra-ROS-Simulation
   ```

2. Place the collected data into "data" directory.

   ```
   $ mv YOUR_DATA ~/Spyndra-Gait-Learning/data
   ```

3. Launch the jupyter notebook from downloaed repository.

   ```
   $ cd ~/Spyndra-Gait-Learning
   $ jupyter notebook
   ``` 

4. If your dataset is generated from a single gait, you can verify repeatibility by running the repeatibility analysis notebook.

   
        
5. If your dataset is generated from random gaits, you can run gait feature notebook to gather global paramter dataset.

## License
This work is licensed under [MIT License](https://opensource.org/licenses/MIT).
