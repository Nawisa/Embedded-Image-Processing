<img src='./images/video_exampe.gif' align="right" width=900>
<br>

# LBP-Based Segmentation of Defocus Blur




## Update2:
Add a python implementation to compute the sharpness metric.

## Update1:
The blur maps for the 1000 images in the [blur segmentation dataset](http://www.cse.cuhk.edu.hk/%7Eleojia/projects/dblurdetect/dataset.html) produced by our algorithm can be found [here](https://1drv.ms/u/s!Aj4IQl4ug0_9hEANDaY7WjuOQkxW) for easy comparison.

##
This repo provides the code to reproduce our defocus segmentaion results  in our [paper](https://ieeexplore.ieee.org/abstract/document/7403985/).
<img src="./images/figure1.png" width="900px"/>

## How to use

### Prerequistites
- Matlab 2016a

If you want to try out the python script, you need

- Python 3
- Opencv 3 (pip install opencv-python)

### Getting Started
```
Note: you can change the features w,h accoring to the real size of the characters of your plate image

All the results reported in the paper were produced by the mex verison of LBP-based sharpness.
```

- To use python script
```

python lbpSharpness.py --input ./test_img/1.jpg
```

### Citations
If you find it useful and are using the code/model/dataset provided here in a publication, please cite our paper:

Yi, Xin, and Mark Eramian. "LBP-based segmentation of defocus blur." IEEE transactions on image processing 25.4 (2016): 1626-1638.

### Acknowlegements
The alpha matting code comes from Levin, 2006:

>A. Levin D. Lischinski and Y. Weiss. *A Closed Form Solution to Natural Image Matting*. 
IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), June 2006, New York.

The multi-scale inference code was adopted form Jianping Shi, 2014:

>Jianping Shi, Li Xu, Jiaya Jia. *Discriminative Blur Detection Features*.
IEEE Conference on Computer Vision and Pattern Recognition, 2014. 









