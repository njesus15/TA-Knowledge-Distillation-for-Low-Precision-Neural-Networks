# Knowledge Distillation of Low Precision Networks with Teacher Assistants
We will take a full precision (32-bit) network and train it on Cifar-100. We will do knowledge distillation to a truncated (24-bit) network. We can either do this with a loss of both networks to the output or with an addition loss of the student to a tempature based of the teacher network.

We will do knowledge distillation from a 32-bit network to a 24-bit network, then use the 24-bit network into a 16-bit network, and repeat with a 8-bit network, and 1-bit network.

We will also train each network individually.

We currently will do the quantization with truncation but use full precision gradients.

We will train on ResNet-20 and SqueezeNet.

## References

* [APPRENTICE: USING KNOWLEDGE DISTILLATION
TECHNIQUES TO IMPROVE LOW-PRECISION NETWORK ACCURACY](https://arxiv.org/pdf/1711.05852.pdf)
* [Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/pdf/1902.03393.pdf)
  * [code](https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation)
* [DOREFA-NET: TRAINING LOW BITWIDTH CONVOLUTIONAL NEURAL NETWORKS WITH LOW BITWIDTH
GRADIENTS](https://arxiv.org/pdf/1606.06160.pdf)
  * [code](https://github.com/bCom5/DoReFa-network-compression)
 * [WRPN: Wide Reduced-Precision Networks](https://arxiv.org/pdf/1709.01134.pdf)
 * [PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085)
 * 

## Todo
* Create the pipeline to train a network and knowledge distillation [Assigned: Brian]
* Figure out truncation or other quantization method. DoReFa or another way to do it. [Assigned: Jesus]
* Figure out transferring the weights or a student loss [Assigned: Aakash]
