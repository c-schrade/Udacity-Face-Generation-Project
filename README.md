# Udacity-Face-Generation-Project

The Jupyter-Notebook-File [dlnd_face_generation.ipynb](dlnd_face_generation.ipynb) includes my
solution to the Face-Generation-Project in the Deeplearning-Nanodegree. 

## General Task and Approach

The task is to construct a neural network that is able to generate images of human faces out
of random vectors.
This task is approached by using a GAN model.

## More precise Description of the Model

As usual, the GAN-architecture consists of two parts: the generator and the discriminator. The
discriminator model is fed real images of human faces and learns to distinguish between real and fake
images of human faces during training. The generator model learns to generate fake images whose quality
is that good that the discriminator struggles to distinguish them from real images. After the
training process the generator is able to produce almost real looking images of human faces with the only
input consisting of 100d random vectors.

The GAN that is trained here is a typical deep convolutional GAN (DCGAN), i.e. the generator essentially
consists of transposed convolutional and the discriminator of convolutional layers. 

All of the code is written in python and the pytorch library is used to implement the DCGAN.

### Hyperparameters

The number of convolutional (resp. transposed convolutional) layers in the discriminator (resp. generator)
is equal to 3 and the GAN is trained for 50 epochs while having a batchsize of 20. Moreover I choose 
the following parameters for the Adam optimizer for the generator and discriminator:

* Learning Rate = 0.002
* Beta1 = 0.5
* Beta2 = 0.999
