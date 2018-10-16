# Representation Learning for Network
## Requirement
- Python 3.6
- PyTorch 0.41 (torch, torchvision)
- Matplotlib
- Numpy
- [myutils](https://github.com/yoshitomo-matsubara/myutils)


## How to clone
```
git clone https://github.com/yoshitomo-matsubara/rep-learn4net.git
cd rep-learn4net/
git submodule init
git submodule update --recursive --remote
```

## How to use
### Quantifying and evaluating a small network
```python src/mnist_runner.py```


### Quantifying multiple small networks
```python src/mnist_plotter.py```
- You can change the input shape by giving **-input** option and its value.  
Format of **-input**: "*channel*,*width*,*height*"  
(default: "1,28,28" for MNIST dataset)
- Same for the ranges of main parameters: the number of channels of the 1st and 2nd 2D convolution layers.  
Format of **-range1** and **-range2**: "*start*:*end*:*step*"  
(default: "5:55:15" and "10:80:20" for the 1st and 2nd layers respectively)