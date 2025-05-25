**Question 1**
Dataloaders `train_loader` and `test_loader` are iterators which allow to access examples and labels.
- What is the type of objects yielded by the train and test dataloaders?
		They yield a tuple with two tensors, one for the images and another with the respective labels, for each batch
- What is the *shape* of `images` returned by `for images, labels in train_loader`? How do you interpret that shape?
		the images will be a tensor of 64 values, one for each pixel
- Why dataloaders for train and test differ with respect to the option `shuffle`?
		The train dataloader must be shuffled after each epoch to prevent overfitting while the test dataloader doesn't need to be shuffled.

**Question 2**
- The model architecture depends on the input data shape. The input images of MNIST are originally of size $28 \times 28$ but they have been resized to $8 \times 8$ pixels. Which changes do you need to do in the script to use size $16 \times 16$? And if you just want to use the original $28 \times 28$ size?
		We would need to change the `transforms.Resize()` command and the `input size` of the model.
- Which change you need to do in the `SimpleNN` class if you want your model to have two hidden layers?
		We would need to define another `nn.Linear()` layer and activation function in the `SimpleNN` class.
- What is a `ReLU` activation function?
		The Rectified Linear Unit outputs a given input if it is positive, otherwise it outputs zero.
- Why are non linear activation functions like `ReLU` necessary for deep learning?
		Non Linear Activation Functions allow the modellation of non-linear relation between inputs and outputs.


**Question 3**
- Which of the following parameters (Image size, Number of nodes in the hidden layer, Loss function, Optimizer) that were defined earlier about the data, model or training you expect will drive computation time down when using GPU? Why?
		A smaller Image Size or Number of nodes in the hidden layer will reduce computation time down when using GPU because the ML would require less computation.


**Question 4**

![[Accuracy Plot.png|500]]**
- From the visualization of that plot, do you think that 5 epochs are enough, or should the model train longer than that?
		The model should train longer because after all epochs the validation accuracy curve still has not converged into a value
- Can you find a reason for the validation curve to be consistently higher than the training curve, which in principle should not happen
		The validation accuracy is higher than train accuracy because the latter is calculated with values before the epoch while the former uses the updated model after the epoch, which yields higher values
