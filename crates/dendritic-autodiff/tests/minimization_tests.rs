
#[cfg(test)]
mod graph_test {

    use std::any::type_name; 
    use dendritic_autodiff::tensor::Tensor; 
    use dendritic_autodiff::node::Node; 
    use dendritic_autodiff::error::{GraphError};
    use dendritic_autodiff::ops::*; 
    use ndarray::prelude::*; 
    use ndarray::{arr2};
    use dendritic_autodiff::graph::{
        ComputationGraph, 
        UnaryOperation, 
        BinaryOperation
    };


    #[test]
    fn test_linear_regression_minimize() {

        let w = Array2::<f64>::zeros((3, 1));
        let b = Array2::<f64>::zeros((1, 1));

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);
        
        let mut graph = ComputationGraph::new();
        graph.mul(x, w); 
        graph.u_add(b); 


    }

}

