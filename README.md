# Harnessing Traditional Controllers for Fast-Track Training of Deep Reinforcement Learning Control Strategies
Achieves path following of an autonomous surface vessel (ASV) with a Deep Deterministic Policy Gradient (DDPG) agent. This code uses reinforcement learning to train an agent to control the rudder angle of the Kriso Container Ship (KCS) to achieve waypoint tracking the presence of calm waters and in the presence of winds

# Why the project is useful?
In recent years, Autonomous Ships have become a focal point for research, with a specific emphasis on improving ship autonomy. Machine Learning Controllers, especially those based on Reinforcement Learning, have seen significant progress. However, addressing the substantial computational demands and intricate reward structures required for their training remains critical. This paper introduces a novel approach, “Leveraging Traditional Controllers for Accelerated Deep Reinforcement Learning (DRL) Training,” aimed at bridging conventional maritime control methods with cutting-edge DRL techniques for vessels. This innovative approach explores the synergies between stable traditional controllers and adaptive DRL methodologies, known for their complexity handling capabilities. To tackle the time-intensive nature of DRL training, we propose a solution: utilizing existing traditional controllers to expedite DRL training by transferring knowledge from these controllers to guide DRL exploration. We rigorously assess the effectiveness of this approach through various ship maneuvering scenarios, including different trajectories and external disturbances like winds. The results unequivocally demonstrate accelerated DRL training while maintaining stringent safety standards. This groundbreaking approach has the potential to bridge the gap between traditional maritime practices and contemporary DRL advancements, facilitating the seamless integration of autonomous systems into maritime operations, with promising implications for enhanced vessel efficiency, cost-effectiveness, and overall safety

# What does the project do?
This repository contains code for implementing reinforcement learning based control for the path following of autonomous ships. The code is trained in a docker container to maintain easy portability.

# How to get started with the project?
The project has been setup with a docker file that can be used to build the docker container in which the code will be executed. It is presumed that you have docker installed on your system. Please follow the following steps to get the code up and running.

**Prerequisites:**

1. An Ubuntu OS (has been tested on Ubuntu 20.04). Note this will not work on Windows OS systems as x11 forwarding needs to be handled separately on it.
2. Docker installed on your system ([Link](https://docs.docker.com/engine/install/ubuntu/))
3. Complete the steps to use docker as a non-root user ([Link](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user))

**Step 1:** 

Clone the repository to your system

```commandline 
git clone https://github.com/Shaadalam9/RL-learning-traditional.git
```

and cd into the src folder. The prompt on the terminal should look as ```.../DDPG-ASV-path-follow/src$```

**Step 2:** 

Change shell scripts to be executables 
```commandline
chmod +x docker.sh
chmod +x run_docker.sh
```

Execute the script docker.sh to build the docker container

```commandline
./docker.sh 
```

**Step 3:** 

Type the following command 

```commandline
xauth list
```
and you will see a response similar (you may see more lines on your computer) to 

```
iit-m/unix:  MIT-MAGIC-COOKIE-1  958745625f2yu22358u3ebe5cc4ad453
#ffff#6969742d6d#:  MIT-MAGIC-COOKIE-1  958745625f2yu22358u3ebe5cc4ad453
```

Copy the first line of the response that corresponds to your computer name (as seen after @ in the prompt on a terminal). So for ```user@iit-m$``` displayed on the prompt the computer name is ```iit-m```.

**Step 4:** 

Run the docker container

```commandline
./run_docker.sh 
```

Notice that this binds the current /src directory on the host machine to the /src directory on the container. Thus, the code can be edited on the host machine and the changes will be reflected in the container instantly. 

**Step 5:** 

Type the following command inside the container (Check that the terminal prompt reflects something like ```docker@iit-m:~/DDPG-ASV-path-follow/src$``` with ```iit-m``` replaced by your computer name)

```commandline
xauth add <your MIT-MAGIC-COOKIE-1>
```
where make sure to replace ```<your MIT-MAGIC-COOKIE-1>``` with your cookie that you copied in step 3. Notice however, that a ```0``` must be added at the end of ```<computer-name/unix:>```

```
xauth add iit-m/unix:0  MIT-MAGIC-COOKIE-1  958745625f2yu22358u3ebe5cc4ad453
```

**Step 6:** 
Now the files should be executable inside the docker container

# Important python scripts

**kcs folder** 

```environment.py``` has the python environment description for the path following of KCS. Edit this file to modify the dynamics of the vehicle and how it interacts with the environment. Currently this study uses the 3-DOF MMG model to mimic the dynamics of the KCS vessel in calm waters and in wind. To start a single training run, edit the training hyperparameters in ```hyperparams.py``` execute the file ```ddpg_train.py``` inside the docker container. Before starting a training run, ensure that the the model and any plots are getting saved in the correct directories. To test various trajectories including single waypoint tracking, elliptical trajectory and other trajectories in calm water or in wind run ```ddpg_test.py``` (make sure to load the correct model). If you want to sequentially train multiple training runs, then set the required hyperparameters in ```batch_train.py``` and execute ```batch_train.py``` inside the docker container.  

# Where can you get help with your project?
Help on this project can be sought by emailing Md. Shadab Alam at `md_shadab_alam@outlook.com` or the supervisor Ignacio Carlucho at `Ignacio.Carlucho@hw.ac.uk`. For any proposed updates to the codes please raise a pull request.

