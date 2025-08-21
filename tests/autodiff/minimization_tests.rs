
#[cfg(test)]
mod graph_test {

    use ndarray::prelude::*; 
    use ndarray::{arr2};

    use dendritic_autodiff::graph::*;
    use dendritic_autodiff::operations::arithmetic::*;
    use dendritic_autodiff::operations::activation::*;
    use dendritic_autodiff::operations::loss::*;

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
        graph.mul(vec![x, w]); 
        graph.add(vec![b]); 

    }

    #[test]
    fn test_mlp_example() {

        let w1 = Array2::<f64>::zeros((2, 3));
        let b1 = Array2::<f64>::zeros((1, 3));
        let w2 = Array2::<f64>::zeros((3, 1));
        let b2 = Array2::<f64>::zeros((1, 1));

        let x = arr2(&[
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]);

        let y = arr2(&[[0.0],[0.0],[1.0],[1.0]]);

        let mut graph = ComputationGraph::new();

        //layer 1
        graph.mul(vec![x, w1]); 
        graph.add(vec![b1]);
        graph.tanh();

        // layer 2
        graph.mul(vec![w2]); 
        graph.add(vec![b2]);
        graph.tanh(); 

        // loss
        graph.mse(y.clone());

        // indicate which nodes are parameters
        graph.add_parameter(1); 
        graph.add_parameter(3); 
        graph.add_parameter(6); 
        graph.add_parameter(8); 

        /*
        for epoch in 0..1000 {
            
            graph.forward();

            graph.backward();

            for var_idx in graph.parameters() {
                let mut var = graph.node(var_idx);
                let grad = var.grad() * (lr as f64);
                let delta = var.output() - grad;
                graph.mut_node_output(var_idx, delta.clone());
            }

        } */

        println!("{:?}", graph.node(10)); 

    }

}

