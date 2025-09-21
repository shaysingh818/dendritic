* Computation graph state needs to be serializable and reused for later computation, essential for saving and loading dynamic model architectures.
* Serialization of shared behavior for each node is required, shared behavior needs to be reconstructed when loaded from saved state
* Longer term snapshots and states of the graph will be saved to have change history of how the graph learns/evolves with training. 

![[serialization-design.jpg]]

## Node Serialization

* Node instance itself requires serializing the shared behavior trait for doing forward and backward operations
* A separate node structure is required for converting fields that can be serialized, specifically the behavior trait needs to map to a string
* The behavior trait is serialized as a string, then it's mapped using an operation registry, the registry maps a string value to the trait.
* The serialized file output is then converted from the structure and the mapping to the string of the trait using the operation registry. 

## Expression Graph Serialization

* The expression graph nodes are serialized into one or many files. All the nodes are kept in its own file partition and each instance utilizes the node serialization routine.
* The extra fields of the expression will be treated as metadata associated with the graph, it will be stored in a separate file partition from the nodes.
* The graphs will contain 2 partitions, one for storing all the nodes, and another for storing the metadata of the graph.