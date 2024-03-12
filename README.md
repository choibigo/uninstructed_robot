## Uninstructed Robot

## quadric slam

#### docker 
```bash
$ cd quadic_slam
$ docker build -t quadricslam:1.0 .
$ docker run -it --name quadricslam --privileged --gpus all --device "/dev:/dev"  quadricslam:1.0 /bin/bash


```