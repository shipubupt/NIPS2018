This is the implementation of our DAT paper "Deep Attentive Tracking via Reciprocative Learning
".      
The project page can be found here:
https://ybsong00.github.io/nips18_tracking/index.     

The pipeline is built upon the py-MDNet tracker for your reference: https://github.com/HyeonseobNam/py-MDNet.   
Note that our DAT tracker does not require offline training using tracking sequences.
# Prerequisites
- GPU: NVIDIA GeForce GTX 1080 Ti
- CUDA 8.0.61
- python 2.7.14
- PyTorch 0.2.0_3 and its dependencies

# Note
If you use our code based on a high-level version of PyTorch for other tasks, please ensure the "retain_graph=True, create_graph=True" in the backward function. Otherwise, the attention map cannot be used to update the parameters.
Thank @Lu Zhou for checking the bug out.

# Usage
1. Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "DAT/models/imagenet-vgg-m.mat"
2. cd DAT/tracking     
   python demo.py


<p>If you find the code useful, please cite both DAT and MDNet:</p>

<pre><code>@inproceedings{nam-cvpr16-MDNET,
    author    = {Nam, Hyeonseob and Han, Bohyung}, 
    title     = {Learning multi-domain convolutional neural networks for visual tracking}, 
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},    
    pages     = {4293--4302},
    year      = {2016},
}
</code></pre>

<pre><code>@inproceedings{shi-nips18-DAT,
    author = {Pu, Shi and Song, Yibing and Ma, Chao and Zhang, Honggang and Yang, Ming-Hsuan},
    title = {Deep Attentive Tracking via Reciprocative Learning},
    booktitle = {Neural Information Processing Systems},
    year = {2018},
  }
</code></pre>




