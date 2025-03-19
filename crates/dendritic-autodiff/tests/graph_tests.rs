
#[cfg(test)]
mod value_test {

    use dendritic_autodiff::graph::{Dendrite, Add2, Sub, Node2}; 
    use ndarray::prelude::*; 
    use ndarray::{arr2};

    #[test]
    fn test_graph_instantiation() {

        let graph: Dendrite<f64> = Dendrite::new();
        assert_eq!(graph.nodes().len(), 0); 

        let node = Node2::new(5.0, 10.0, Box::new(Add2)); 

    }

}
