# Trainable Trait

* The trainable trait is a reusable interface that models can implement to train samples, or train multiple batches of samples on. 
* The trainable trait does not have functionality to work with optimizer, the `TrainOptimizer` trait should be used if use of optimizer while training is desired. 
## Train
| Argument Name | Type    | Description                                            |
| ------------- | ------- | ------------------------------------------------------ |
| `epochs`      | `usize` | Number of epochs to train until convergence is reached |

* Trains and iterates on the whole dataset provided for a model, does not make use of batching or stochastic methods. 
## Train Batch
| Argument Name  | Type    | Description                                                        |
| -------------- | ------- | ------------------------------------------------------------------ |
| `iterations`   | `usize` | Number of iterations to train batches of data on.                  |
| `batch_size`   | `usize` | Batch size to slice and train data on from whole dataset           |
| `batch_epochs` | `usize` | Number of epochs to iterate on each batched training per iteration |

* Makes use of batch gradient descent & random shuffling for training larger datasets. Generates loading and progress for each batch with loss convergence results. 

# Train Optimizer Trait

* The train optimizer trait is a resuable interface that models can inherit from to implement training logic with an optimizer. 
* The train optimizer trait takes in an instance of a trait that implements an optimizer. The optimizer will be called in the training methods to update parameters. 

## Train With Optimizer
| Argument Name | Type        | Description                               |
| ------------- | ----------- | ----------------------------------------- |
| `epochs`      | `usize`     | Number of iterations to train dataset on. |
| `optimizer`   | `Optimizer` | Instance of optimizer trait               |

* Trains and updates parameters on a dataset using an optimizer as a parameter. This method can take in any structure that implements the optimizer shared behavior trait. 

## Train Batch With Optimizer
| Argument Name  | Type        | Description                                                        |
| -------------- | ----------- | ------------------------------------------------------------------ |
| `iterations`   | `usize`     | Number of iterations to train batches of data on.                  |
| `batch_size`   | `usize`     | Batch size to slice and train data on from whole dataset           |
| `batch_epochs` | `usize`     | Number of epochs to iterate on each batched training per iteration |
| `optimizer`    | `Optimizer` | Instance of optimizer trait                                        |

* Trains and updates parameters for batches of data that are fed through the training loop, optimizer is passed in as parameter and is used for updating parameters.