# Knowledge Distillation of Low Precision Networks with Teacher Assistants
We will take a full precision (32-bit) network and train it on Cifar-100. We will do knowledge distillation to a truncated (16-bit) network. We can either do this with a loss of both networks to the output or with an addition loss of the student to a tempature based of the teacher network.

We will do knowledge distillation from a 32-bit network to a 16-bit network, then use the 16-bit network into a 8-bit network, and repeat with a 4-bit network, 2-bit, and 1-bit network.

We will also train each network individually and compare the TA method vs. individually.

We currently will do the quantization with truncation but use full precision gradients. We will not change parameter size or model depth.

We will train on ResNet-20 and possibly SqueezeNet.

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
 * [Quantizing deep convolutional networks for efficient inference: A whitepaper] (https://arxiv.org/pdf/1806.08342.pdf)

## Todo
* Create the pipeline to train a network and knowledge distillation [Assigned: Brian]
* Figure out truncation or other quantization method. DoReFa or another way to do it. [Assigned: Jesus]
* Figure out transferring the weights or a student loss [Assigned: Aakash]

### Update 4/13/20
* Jesus - finish quantization methods
* Updated README for what we plan on doing.

### Update 4/15/20
* (Jesus) Quantization should work now. Simply call get_quant_model(resnet_model, qparams) from resent_quant.py
where qparams is a tuple [wbit, qbit, 'dorefa']. Only have dorefa implemented although other methods mentioned 
above are very similar and should be easy to add. Since computations are still done full precision, I beleive
calling .backward() on the loss computes full precision gradients.

Short example on ResNetQ_Example.ipynb notebook prints forward pass outputs which include quantized activations
and weights.

