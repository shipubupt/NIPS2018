This is the implementation of our DAT paper "Deep Attentive Tracking via Reciprocative Learning
".      
The project page can be found here:
https://ybsong00.github.io/nips18_tracking/index.     

The pipeline is built upon the py-MDNet tracker for your reference: https://github.com/HyeonseobNam/py-MDNet.   
Note that our DAT tracker does not require offline training using tracking sequences.

##Usage
<pre><code>
1. Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
2. cd tracking
   python demo.py
</code></pre>

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




