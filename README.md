# Torch-VDSR
Torch implementation of the VDSR-CNN Upscaling algorithm

This is a simple torch implementation of the VDSR algorithm from this paper, though some hyperparameters were modified for faster convergence.  
https://arxiv.org/abs/1511.04587


Included model uses only 8 layers (instead of the paper's 20), was trained for 2 hours on a R9 290x using 7 high-resolution images.  

This is a preliminary test so the results are not the best. When I have time I will train the full 20 layer model using a bigger dataset (around 1000 images).  

![miku comparison](https://github.com/bloc97/Torch-VDSR/raw/master/demo/miku_comp.png)
![bird comparison](https://github.com/bloc97/Torch-VDSR/raw/master/demo/bird_comp.png ) 
