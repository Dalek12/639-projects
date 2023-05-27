from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
import torch.nn as nn

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of random noise from Gaussian distribution.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing
      noise from a Gaussian distribution.
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    noise = torch.randn(batch_size, noise_dim, dtype=dtype, device=device)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Flatten(),
        # Fully connected layer with input size 784 and output size 400
        nn.Linear(784, 400),
        # LeakyReLU with alpha 0.05
        nn.LeakyReLU(0.05),
        # Fully connected layer with input size 400 and output size 200
        nn.Linear(400, 200),
        # LeakyReLU with alpha 0.05
        nn.LeakyReLU(0.05),
        # Fully connected layer with input size 200 and output size 100
        nn.Linear(200, 100),
        # LeakyReLU with alpha 0.05
        nn.LeakyReLU(0.05),
        # Fully connected layer with input size 100 and output size 1
        nn.Linear(100, 1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        # Fully connected layer with input size noise_dim and output size 128
        nn.Linear(noise_dim, 128),
        # ReLU
        nn.ReLU(),
        # Fully connected layer with input size 128 and output size 256
        nn.Linear(128, 256),
        # ReLU
        nn.ReLU(),
        # Fully connected layer with input size 256 and output size 512
        nn.Linear(256, 512),
        # ReLU
        nn.ReLU(),
        # Fully connected layer with input size 512 and output size 784
        nn.Linear(512, 784),
        # TanH (to clip the image to be in the range of [-1, 1])
        nn.Tanh()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    true_labels = torch.ones_like(logits_real, device=logits_real.device)
    fake_labels = torch.zeros_like(logits_fake, device=logits_fake.device)

    real_loss = F.binary_cross_entropy_with_logits(logits_real, true_labels)
    fake_loss = F.binary_cross_entropy_with_logits(logits_fake, fake_labels)
    loss = real_loss + fake_loss

    return loss
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code

    fake_labels = torch.ones_like(logits_fake, device=logits_fake.device)
    loss = F.binary_cross_entropy_with_logits(logits_fake, fake_labels)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def run_a_gan(D, G, D_solver, G_solver, loader_train, discriminator_loss, generator_loss, device, show_images, plt, show_every=250, 
              batch_size=128, noise_size=96, num_epochs=10):
  """
  Train a GAN!
  
  Inputs:
  - D, G: PyTorch models for the discriminator and generator
  - D_solver, G_solver: torch.optim Optimizers to use for training the
    discriminator and generator.
  - loader_train: the dataset used to train GAN
  - discriminator_loss, generator_loss: Functions to use for computing the generator and
    discriminator loss, respectively.
  - show_every: Show samples after every show_every iterations.
  - batch_size: Batch size to use for training.
  - noise_size: Dimension of the noise to use as input to the generator.
  - num_epochs: Number of epochs over the training dataset to use for training.
  """
  iter_count = 0
  for epoch in range(num_epochs):
    for x, _ in loader_train:
      if len(x) != batch_size:
        continue
      ##############################################################################
      # TODO: Implement an iteration of training the discriminator.                #
      # Replace 'pass' with your code.                                             #
      # Save the overall discriminator loss in the variable 'd_total_error',       #
      # which will be printed after every 'show_every' iterations.                 #
      #                                                                            #    
      # IMPORTANT: make sure to pre-process your real data (real images),          #
      # so as to make it in the range [-1,1].                                      #
      ##############################################################################
      d_total_error = None
      D_solver.zero_grad()
      real_images = x.view(-1, 784).to(device)
      logits_real = D(2* (real_images - 0.5))
      #print("Real images dtype:", real_images.dtype)
      #print("Real images shape:", real_images.shape)

      # Train the discriminator
      noise = sample_noise(batch_size, noise_size, dtype=real_images.dtype, device=real_images.device)
      fake_images = G(noise).detach()
      #print("Fake images dtype:", fake_images.dtype)
      #print("Fake images shape:", fake_images.shape)
      logits_fake = D(fake_images)
      d_total_error = discriminator_loss(logits_real, logits_fake)
      d_total_error.backward()
      D_solver.step()
      ##############################################################################
      #                              END OF YOUR CODE                              #
      ##############################################################################        


      ##############################################################################
      # TODO: In the same iteration, implement training of the generator now   .   #
      # Replace 'pass' with your code.                                             #
      # Save the generator loss in the variable 'g_error', which will be printed.  #
      # after every 'show_every' iterations, and save the fake images generated    #
      # by G in the variable 'fake_images', which will be used to visualize the    #
      # generated images.
      ##############################################################################
      g_error = None
      fake_images = None
      g_noise = sample_noise(batch_size, noise_size, dtype=real_images.dtype, device=real_images.device)
      fake_images = G(g_noise)
      logits_fake = D(fake_images)
      g_error = generator_loss(logits_fake)
      G_solver.zero_grad()
      g_error.backward()
      G_solver.step()

      ##############################################################################
      #                              END OF YOUR CODE                              #
      ############################################################################## 
  
      if (iter_count % show_every == 0):
        print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
        imgs_numpy = fake_images.data.cpu()#.numpy()
        show_images(imgs_numpy[0:16])
        plt.show()
        print()
      iter_count += 1
    if epoch == num_epochs - 1:
      return imgs_numpy    


#class PrintShape(nn.Module):
    #def forward(self, x):
      #  print("Discriminator input shape:", x.shape)
       # return x

def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        # Reshape into image tensor (Use nn.Unflatten!)
        #PrintShape(),
        nn.Unflatten(1, (1, 28, 28)),
        
        # Conv2D: 32 Filters, 5x5, Stride 1
        nn.Conv2d(1, 32, kernel_size=5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, stride=2),
        
        # Conv2D: 64 Filters, 5x5, Stride 1
        nn.Conv2d(32, 64, kernel_size=5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, stride=2),
        
        # Flatten
        nn.Flatten(),
        
        # Fully Connected with output size 4 x 4 x 64
        nn.Linear(4 * 4 * 64, 1024),
        nn.LeakyReLU(0.01),
        
        # Fully Connected with output size 1
        nn.Linear(1024, 1)
    )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        # Fully connected with output size 1024
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        
        # Fully connected with output size 7 x 7 x 128
        nn.Linear(1024, 7 * 7 * 128),
        nn.ReLU(),
        nn.BatchNorm1d(7 * 7 * 128),
        
        # Reshape into Image Tensor of shape 7 x 7 x 128
        nn.Unflatten(1, (128, 7, 7)),
        
        # Conv2D^T (Transpose): 64 filters of 4x4, stride 2, 'same' padding (use padding=1)
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        
        # Conv2D^T (Transpose): 1 filter of 4x4, stride 2, 'same' padding (use padding=1)
        nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        
        # Should have a 28 x 28 x 1 image, reshape back into 784 vector
        nn.Flatten()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
