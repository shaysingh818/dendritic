# Optimizer

* Retrieve all nodes in the computation graph that are marked as a parameter. 
* Each optimizer type keeps a value associated with a parameter that could either be the accumulation of updates or velocity per parameter

# Optimizer Trait

* Optimizer trait allows extending of existing implementation that update parameters using different techniques.
* Optimizer trait updates parameters of graph associated with `Model` trait. Trait uses an instance of a model trait as a parameter for optimization.
## Step
| Argument Name | Type         | Description                     |
| ------------- | ------------ | ------------------------------- |
| `model`       | `&mut Model` | Instance of mutable model trait |

* Controls how parameter are updated for a given model, uses trait methods in `Model` to update parameters with access to the graph. 

## Default Optimizer's
| Optimizer Name | Description                                                                           |
| -------------- | ------------------------------------------------------------------------------------- |
| **Nesterov**   | Extension/modification of momentum based algorithm for faster convergence             |
| **Adagrad**    | Optimizer that adapts a learning rate for each design point.                          |
| **RMSProp**    | Extension of `Adagrad` to avoid effect of decreasing learning rate.                   |
| **Adadelta**   | Also extension of `Adagrad` to avoid effect of continuously decreasing learning rate. |
| **Adam**       | Optimization technique that adapts a learning rate to each parameter                  |



