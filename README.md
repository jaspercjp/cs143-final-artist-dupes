# Three Flavors of Neural Style Transfer
![poster](./results/artist-dupes-style-transfer.svg)

## Project Description
The fusion of technology and art has opened up unprecedented avenues for creative expression, particularly through image style transfer. This technique was first introduced by Gatys and colleagues in their paper, "A Neural Algorithm of Artistic Style"\cite{gatys2015neural}, which utilizes pre-trained convolutional neural networks (CNNs) to blend the content of one image with the stylistic elements of another. Artistic movements like Impressionism, Surrealism, and Pop Art each emerged in response to the cultural and societal shifts of their time, offering new perspectives and expressions. With the new digital age, the ability to digitally reimagine images in the styles of such movements bridges a gap between the past and the present. Thus, our project aims to explore ways to make art more accessible and inspire new forms of expression that honors historical art movements. In this project, we expand upon the foundational techniques developed by Gatys et al, delving into various optimizations and methodologies for performing style transfer more efficiently -- exploring patch-based approaches and feedforward networks. Our goal is to assess these methods not only for their aesthetic output but also for their efficiency and practical applicability. 

### [Gatys et al:](https://arxiv.org/pdf/1508.06576)
In their paper, Gatys et al. sought out to combine the semantic meaning of a content image with the style of a style image. Using the pre-trained VGG16 CNN, they were able to utilize the activations at different layers as proxies of high-level content and style informations. They then utilized a loss function, on which an image was optimized to produce both similar content and style responses when fed into the pretrained network to accomplish style transfer. 

### [Johnson et al.](https://arxiv.org/pdf/1603.08155)
To expand on the ideas presented in Gatys et al., Johnson et al. sought to define a feed-forward neural network that essentially eliminated the need for the optimization of the loss function defined in Gatys et al. Essentially, the feed-forward neural network would learn the style of the style image given, and having learned the correct filters and down/upscaling parameters, it would then auto encode a content image to be in the style of the learned style. This cut down the time needed to produce a transformed image, instead replacing such a time cost with a training cost. 

### [Chen et al](https://arxiv.org/pdf/1612.04337) 
To expand on the ideas presented in Gatys et al, Chen et al. utilized the VGG19 model to extract feature maps from both the style and content images through a single layer. The core of this method is seen in the style swap function, where each patch of a defined size is extracted from the style feature maps, and is slid over the style features with a specified stride. To identify the best matching patches, the content features undergo a convolution operation with the normalized style patches to assess the compatibility between each patch and local regions in the content features. Then, the best matching patches are reconstructed into a style-transformed feature map with the spatial dimensions of the original content features through a transposed convolution operation. This transformation is further refined through an optimization process, where a loss function is minimized through style loss and total variation regularization.


## Contributors:
* Jasper Chen
* Rachel Chae
* Sean Yu
