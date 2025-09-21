# Automatic Differentiation

Create differentiable expressions using the existing rust operators with custom defined operations. Framework for experimenting with ways to minimize or maximize functions with different combinations of operations/activation functions. Developed expressions can be stored in a computation/expression graph that stores the state to avoid doing the same re computations. Specifically to our optimization library, instead of calling a generic graph a "graph", we will call it a "dendrite", similar to how pytorch calls theirs a "torch"

# Features
| Feature Alias           | Description                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| **Expression graph**    | Computation graph structure to store operations and nodes in a differentiable expression. |
| **Operation Extension** | Extensible operations module for creating custom operations for auto differentiation      |
| **Serialization**       | Serialization layer for saving and loading expressions                                    |

# Supported Operations

Supported default operations that come with dendritic, a  user does not need to extend any traits to use these methods. Default support operations are in the operation registry so they can be looked up for serialization & later use. 

## Arithmetic
| Operation Alias | Types Supported      | Description                                        |
| --------------- | -------------------- | -------------------------------------------------- |
| `Add`           | `Array2<f64>`, `f64` | Adds one or more values (binary & unary)           |
| `Sub`           | `Array2<f64>`, `f64` | Subtracts one or more values (binary & unary)      |
| `Mul`           | `Array2<f64>`, `f64` | Multiplication of one of more values (dot product) |

## Activation
| Operation Alias | Types Supported      | Description                                                  |
| --------------- | -------------------- | ------------------------------------------------------------ |
| `Sigmoid`       | `Array2<f64>`, `f64` | Performs sigmoid activation function on inputs to operation. |
| `Tanh`          | `Array2<f64>`, `f64` | Performs tanh activation function on inputs to operation.    |

## Loss
| Operation Alias           | Types Supported      | Description                                                                                          |
| ------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `MSE`                     | `Array2<f64>`, `f64` | Mean squared error loss function                                                                     |
| `BinaryCrossEntropy`      | `Array2<f64>`, `f64` | Binary cross entropy loss function for classification                                                |
| `CategoricalCrossEntropy` | `Array2<f64>`, `f64` | Categorical cross entropy loss function for multiclass classification                                |
| `DefaultLossFunction`     | `Array2<f64>`, `f64` | Default loss function, purpose is for model archtecture prototypes without an unknown loss function. |

