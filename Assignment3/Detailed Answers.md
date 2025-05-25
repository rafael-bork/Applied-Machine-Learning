**Question 1**
Dataloaders `train_loader` and `test_loader` are iterators which allow to access examples and labels.
- What is the type of objects yielded by the train and test dataloaders?
		The train_loader and test_loader serve as iterators, allowing us to analyse the dataset in batches for each epoch, making the process mor efficient and less time-consuming. At each iteration, they yield a tuple with two tensors one with the images and another with the respective labels.
- What is the *shape* of `images` returned by `for images, labels in train_loader`? How do you interpret that shape?
		Each image in a batch has been resized to 8×8 pixels and then flattened into a tensor with 64 values, each value representing the intensity of a pixel in the original image.
- Why dataloaders for train and test differ with respect to the option `shuffle`?
		For the training loader, shuffling ensures that the dataset is presented in a differently order at each epoch. This is very important for the model to avoid learning the order of data and being overfitted. In contrast, the test data doesn't affect the model's fitting and ensures consistent evaluation of the model across epochs, therefore, there's no need to shuffle the test data  


**Question 2**
- The model architecture depends on the input data shape. The input images of MNIST are originally of size $28 \times 28$ but they have been resized to $8 \times 8$ pixels. Which changes do you need to do in the script to use size $16 \times 16$? And if you just want to use the original $28 \times 28$ size?
		The architecture of the neural network model depends fundamentally on the size of the input data. The MNIST images originally have a resolution of 28×28 pixels, but in the given script, they are resized to 8×8. If we want to change the input image size from 8x8 pixels, to 16×16 pixels, we would need to update the transforms.Resize() function to reflect the new dimensions. Since the images are flattened before being inputted into the model, their input must also be changed to accept this new 256 vector. Similarly, if we want to use the original 28×28 size, the input size would need to be updated to 784.
- Which change you need to do in the `SimpleNN` class if you want your model to have two hidden layers?
		To modify the SimpleNN class so that the model includes two hidden layers instead of one, we would need to add an additional `nn.Linear()` layer in the constructor and add another non-linear activation function in the forward() method.
- What is a `ReLU` activation function?
		The Rectified Linear Unit activation function, outputs zero for negative inputs and returns the input itself for positive values.
- Why are non linear activation functions like `ReLU` necessary for deep learning?
		Non Linear Activation Functions introducesnon-linearity into the model, which is essential for the model to learn complex patterns.

**Question 3**
- Which of the following parameters (Image size, Number of nodes in the hidden layer, Loss function, Optimizer) that were defined earlier about the data, model or training you expect will drive computation time down when using GPU? Why?
		Among the parameters defined, the image size and the number of nodes in the hidden layer are the ones that most significantly impact computation time on a GPU, because both influence the number of calculations the model must perform. A smaller image size leads to fewer input features, and fewer nodes in the hidden layer reduce the number of weights and operations required. In contrast, choices like the loss function or the optimizer do not change in a meaningful way the number of computations necessary.


**Question 4**
- From the visualization of that plot, do you think that 5 epochs are enough, or should the model train longer than that?
		Looking at the training and validation accuracy curves, we can admit that after all epochs the validation accuracy is still increasing and has not yet plateaued, this suggests that the model has not converged, and further training could continue to improve its performance. This way we can admit that the numbers of epochs for training should be extended.
- Can you find a reason for the validation curve to be consistently higher than the training curve, which in principle should not happen
		Curiously, it appears that the validation accuracy is always higher than the training accuracy, which is counterintuitive since we expect the model to perform better on the training data it had already seen. This can be explained by how the accuracies are computed. Usually training accuracy is measured after the model has fully updated its weights at the end of the epoch, which is not happening, it is being calculated before that, and the validation accuracy is being computed using the fully updated model. This means that we are comparing the performance a outdated model with an improved one, for the same epoch, which can make the validation accuracy higher than the training accuracy . 
