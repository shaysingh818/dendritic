# Operation Extension Design

* Operation extension design for how users can extend and create their own custom operations for automatic differentiation
* User extends behavior trait that is attached to node instance, behavior trait contains methods for the forward and backward pass.
* Behavior trait is attached to node instance which is part of the larger expression/computation graph.
* User has access to other nodes in the graph along with current node index when extending/creating custom operations.

![Test](../assets/Operation%20Extension/dendritic-operation-extension.jpg)

# Behavior Trait

* Interface that captures abstract but common methods needed for doing a forward and backward pass in the computation graph
* Trait is attached to node instances associated with the computation graph. Behavior trait methods are called generically
* Trait is generic, user can extend the trait and provide a type that the forward and backward pass parameters should accept

## Forward
| Parameter       | Parameter Type | Description                                                                             |
| --------------- | -------------- | --------------------------------------------------------------------------------------- |
| `nodes`         | `Vec<Node<T>>` | Vector of nodes that are populated in computation graph. Takes in a generic type        |
| `curr_node_idx` | `usize`        | Current node index where computation is being performed in the graph using forward pass |

* User depends on the `curr_node_idx` parameter and refrences the node by that index and retrieves the inputs
* Operation is computed in forward pass using the inputs for the `curr_node_idx`, input indices are used to access other `nodes` in the graph

## Backward
| Parameter       | Parameter Type | Description                                                                                                           |
| --------------- | -------------- | --------------------------------------------------------------------------------------------------------------------- |
| `nodes`         | `Vec<Node<T>>` | Vector of nodes that are populated in computation graph. Backward pass has access to all nodes currently in the graph |
| `curr_node_idx` | `usize`        | Current node index where computation is being performed in the graph using backward pass                              |

* User retrieves the inputs associated with the `curr_node_idx` that is being used to access the current node in the backward computation
* Upstream node indexes are also retrieved with the current node index, upstream indices are used to pass upstream gradients downstream
* Upstream gradients are factored into gradient calculation, gradients of inputs are then updated based on upstream values