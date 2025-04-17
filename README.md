# About 

This is code for my [Bachelor thesis](https://dspace.cvut.cz/handle/10467/94751) in Cybernetics and Robotics course at [Faculty of Electrical Engineering](https://fel.cvut.cz/cs) at [Czech Technical University in Prague](https://www.cvut.cz/en/).

The main goal was to capture images of terrain via quadcopter and from them recostruct a 3D model which can be used for navigation and obstacle avoidance near the surface. Since there were alrady numerous open-source photogrammetry implementations out there, I decided to compare them in noisy and noise-free conditions. In order to have reliable ground-truth terrain model, I used the [AirSim](https://microsoft.github.io/AirSim/) Unreal Engine plugin with a custom modelled 3D environment. 

<div align="center">
<img src="https://github.com/user-attachments/assets/fbeaabbe-3e2b-4d67-ac2f-3c7026480c0f" alt="Description" width="480" height="240"/>
</div>

The images from the simulator were processed using 3 different photogrammetry pipelines to turn them into 3D models and also 2.5D heightmaps used for navigation near the surface. The left image here shows the ground truth 3D model of the terrain, whereas the right image shows the model recovered from the images.

<div align="center">
<img src="https://github.com/user-attachments/assets/e4798edf-11ec-49b1-b1b5-cf007802f843" alt="CloudGT" height="240"/>
<img src="https://github.com/user-attachments/assets/60755f9e-c567-41dd-8a61-a5e2291abda3" alt="CloudColmap" height="240"/>
</div>

The 2.5D heightmap is then used to define obstacles and costs for the A* algorithm, which outputs an optimal path through the environment.

<div align="center">
<img src="https://github.com/user-attachments/assets/066a7bb8-ab4f-4a7f-9d13-9258728d0a3b" alt="Description" width="480" height="240"/>
</div>

Video:
<p align="center"><a href="https://www.youtube.com/watch?v=AKZbAtYz784">
<img src="http://img.youtube.com/vi/AKZbAtYz784/0.jpg" alt="Video">
</a></p>
