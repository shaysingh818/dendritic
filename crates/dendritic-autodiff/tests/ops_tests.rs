#[cfg(test)]
mod operations_test {

    use dendritic_autodiff::node::{Node};
    use dendritic_autodiff::graph::{ComputationGraph};
    use dendritic_autodiff::operations::activation::*; 
    use dendritic_autodiff::operations::arithmetic::*; 
    use dendritic_autodiff::operations::loss::*; 
    use ndarray::prelude::*; 
    use ndarray::{arr2};

    #[test]
    fn test_add() {

        let mut scalar_graph = ComputationGraph::new();
        scalar_graph.add(vec![2.0, 3.0]);
        scalar_graph.add(vec![4.0]);
    
        assert_eq!(scalar_graph.nodes().len(), 5); 

        assert_eq!(scalar_graph.node(0).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(0).output(), 2.0);
        assert_eq!(scalar_graph.node(0).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(1).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(1).output(), 3.0); 
        assert_eq!(scalar_graph.node(1).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(2).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(2).output(), 0.0);
        assert_eq!(scalar_graph.node(2).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(3).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(3).output(), 4.0);
        assert_eq!(scalar_graph.node(3).upstream(), vec![4]);
 
        assert_eq!(scalar_graph.node(4).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(4).output(), 0.0);
        assert_eq!(scalar_graph.node(4).upstream().len(), 0);

        scalar_graph.forward();

        assert_eq!(scalar_graph.node(2).output(), 5.0); 
        assert_eq!(scalar_graph.node(4).output(), 9.0);

        scalar_graph.backward();

        assert_eq!(scalar_graph.node(3).grad(), 1.0); 
        assert_eq!(scalar_graph.node(2).grad(), 1.0); 
        assert_eq!(scalar_graph.node(1).grad(), 1.0); 
        assert_eq!(scalar_graph.node(0).grad(), 1.0); 

        let a = arr2(&[[1.0], [2.0], [3.0]]); 
        let b = arr2(&[[1.0], [2.0], [3.0]]); 
        let c = arr2(&[[1.0], [1.0], [1.0]]); 

        let mut nd_graph = ComputationGraph::new();
        nd_graph.add(vec![a.clone(), b.clone()]);
        nd_graph.add(vec![c.clone()]);
        nd_graph.default(); 

        assert_eq!(nd_graph.nodes().len(), 6); 
        assert_eq!(nd_graph.path().len(), 0); 

        nd_graph.forward();

        assert_eq!(nd_graph.node(2).output().shape(), vec![3, 1]); 
        assert_eq!(
            nd_graph.node(2).output(),
            arr2(&[[2.0],[4.0],[6.0]])
        );

        assert_eq!(nd_graph.node(4).output().shape(), vec![3, 1]); 
        assert_eq!(
            nd_graph.node(4).output(),
            arr2(&[[3.0],[5.0],[7.0]])
        );

        nd_graph.backward();

        let vars = nd_graph.variables();
        let expected_grads = vec![a, b, c];

        for (idx, var) in vars.iter().enumerate() {
            println!("{:?}", nd_graph.node(*var).grad()); 
            /*
            assert_eq!(
                nd_graph.node(*var).grad(),
                expected_grads[idx]
            ); */
        } 
    }

    #[test]
    fn test_subtract() {

        let mut scalar_graph = ComputationGraph::new();
        scalar_graph.sub(vec![10.0, 5.0]);
        scalar_graph.sub(vec![2.0]);
    
        assert_eq!(scalar_graph.nodes().len(), 5); 

        assert_eq!(scalar_graph.node(0).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(0).output(), 10.0);
        assert_eq!(scalar_graph.node(0).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(1).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(1).output(), 5.0); 
        assert_eq!(scalar_graph.node(1).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(2).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(2).output(), 0.0);
        assert_eq!(scalar_graph.node(2).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(3).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(3).output(), 2.0);
        assert_eq!(scalar_graph.node(3).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(4).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(4).output(), 0.0);
        assert_eq!(scalar_graph.node(4).upstream().len(), 0);

        scalar_graph.forward();

        assert_eq!(scalar_graph.node(2).output(), 5.0); 
        assert_eq!(scalar_graph.node(4).output(), 3.0);

        scalar_graph.backward();

        let vars = scalar_graph.variables();
        for (idx, var) in vars.iter().enumerate() {
            assert_eq!(scalar_graph.node(*var).grad(), 1.0);
        }

    } 

    #[test]
    fn test_multiply() {

        let mut scalar_graph = ComputationGraph::new();
        scalar_graph.mul(vec![10.0, 5.0]);
        scalar_graph.mul(vec![2.0]);
    
        assert_eq!(scalar_graph.nodes().len(), 5); 

        assert_eq!(scalar_graph.node(0).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(0).output(), 10.0);
        assert_eq!(scalar_graph.node(0).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(1).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(1).output(), 5.0); 
        assert_eq!(scalar_graph.node(1).upstream(), vec![2]);

        assert_eq!(scalar_graph.node(2).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(2).output(), 0.0);
        assert_eq!(scalar_graph.node(2).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(3).inputs().len(), 0); 
        assert_eq!(scalar_graph.node(3).output(), 2.0);
        assert_eq!(scalar_graph.node(3).upstream(), vec![4]);

        assert_eq!(scalar_graph.node(4).inputs().len(), 2); 
        assert_eq!(scalar_graph.node(4).output(), 0.0);
        assert_eq!(scalar_graph.node(4).upstream().len(), 0);

        scalar_graph.forward();

        assert_eq!(scalar_graph.node(2).output(), 50.0); 
        assert_eq!(scalar_graph.node(4).output(), 100.0);

        scalar_graph.backward();

        let a = arr2(&[
            [1.0, 1.0, 1.0], 
            [2.0, 2.0, 2.0], 
            [3.0, 3.0, 3.0]
        ]); 
        let b = arr2(&[[1.0], [2.0], [3.0]]); 
        let c = arr2(&[[1.0]]); 

        let mut nd_graph = ComputationGraph::new();
        nd_graph.mul(vec![a.clone(), b.clone()]);
        nd_graph.mul(vec![c.clone()]);
        nd_graph.default(); 

        assert_eq!(nd_graph.nodes().len(), 6); 
        assert_eq!(nd_graph.path().len(), 0); 

        nd_graph.forward();

        assert_eq!(nd_graph.node(2).output().shape(), vec![3, 1]); 
        
        assert_eq!(
            nd_graph.node(2).output(),
            arr2(&[[6.0],[12.0],[18.0]])
        );

        assert_eq!(nd_graph.node(4).output().shape(), vec![3, 1]); 
        assert_eq!(
            nd_graph.node(4).output(),
            arr2(&[[6.0],[12.0],[18.0]])
        );

        nd_graph.backward(); 

        let grad_1 = arr2(&[
            [6.0, 12.0, 18.0],
            [12.0, 24.0, 36.0],
            [18.0, 36.0, 54.0]
        ]);
        let grad_2 = arr2(&[[84.0], [84.0], [84.0]]);
        let grad_3 = arr2(&[[504.0]]);
        let grads = vec![grad_1, grad_2, grad_3]; 

        let vars = nd_graph.variables();
        for (idx, var) in vars.iter().enumerate() {
            assert_eq!(
                nd_graph.node(*var).grad(),
                grads[idx]
            );
        }
    }

    #[test]
    fn test_mse() {

        let a = arr2(&[[1.0], [2.0], [3.0]]); 
        let b = arr2(&[[1.0], [2.0], [3.0]]); 
        let c = arr2(&[[1.0], [1.0], [1.0]]); 

        let mut graph = ComputationGraph::new();
        graph.add(vec![a, b]);
        graph.mse(c);

        assert_eq!(graph.nodes().len(), 5);

        assert_eq!(graph.node(0).inputs().len(), 0); 
        assert_eq!(graph.node(0).output().shape(), vec![3, 1]); 

        assert_eq!(graph.node(1).inputs().len(), 0); 
        assert_eq!(graph.node(1).output().shape(), vec![3, 1]); 

        assert_eq!(graph.node(2).inputs().len(), 2); 
        assert_eq!(graph.node(2).output().shape(), vec![0, 0]); 

        graph.forward();

        assert_eq!(graph.node(2).output().shape(), vec![3, 1]); 
        assert_eq!(
            graph.node(2).output(), 
            arr2(&[[2.0],[4.0],[6.0]])
        );

        graph.backward();

        assert_eq!(
            graph.node(2).grad(), 
            arr2(&[[1.0],[3.0],[5.0]])
        );

    }


    #[test]
    fn test_binary_cross_entropy() {

        let a = arr2(&[
            [0.0], [0.0], [1.0], [0.0], [1.0],
            [1.0], [1.0], [1.0], [1.0], [1.0]
        ]);


        let b = arr2(&[
            [0.0], [0.0], [0.0], [0.0], [0.0],
            [0.0], [0.0], [0.0], [0.0], [0.0]
        ]);

        let c = arr2(&[
            [0.19], [0.33], [0.47], [0.7], [0.74],
            [0.81], [0.86], [0.94], [0.97], [0.99]
        ]); 

        let mut graph = ComputationGraph::new();
        graph.add(vec![c, b]);
        graph.bce(a);

        assert_eq!(graph.nodes().len(), 5); 

        graph.forward();

        let output = graph.curr_node();
        let output_nd = output.output();
        let output_val = output_nd[[0, 0]];

        assert_eq!(output_val, 3.335227947407202); 
        assert_eq!(output_nd.shape(), vec![1, 1]);

        graph.backward();

        assert_eq!(graph.node(4).grad().len(), 10); 

    }


    #[test]
    fn test_sigmoid() {

        let lr: f64 = 0.02; 
        let w = Array2::<f64>::zeros((3, 1));
        let b = Array2::<f64>::zeros((1, 1));

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]); 
        
        let mut graph = ComputationGraph::new();
        graph.mul(vec![x, w]); 
        graph.add(vec![b]);
        graph.sigmoid(); 
        graph.mse(y.clone());

        graph.forward(); 

        let add_output = graph.node(4); 
        let sig_output = graph.node(5); 

        assert_eq!(
            add_output.output(), 
            arr2(&[[0.0],[0.0],[0.0],[0.0], [0.0]])
        );

        assert_eq!(
            sig_output.output(), 
            arr2(&[[0.5],[0.5],[0.5],[0.5], [0.5]])
        );

        graph.backward();

        assert_eq!(
            graph.node(4).grad(),
            arr2(&[[-9.5],[-11.5],[-13.5],[-15.5],[-17.5]])
        );

    }


    #[test]
    fn test_tanh() {

        let lr: f64 = 0.02; 
        let b = Array2::<f64>::zeros((1, 1));
        let w = arr2(&[[1.0],[2.0],[3.0]]);

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]); 
        
        let mut graph = ComputationGraph::new();
        graph.mul(vec![x, w]); 
        graph.add(vec![b]);
        graph.tanh(); 
        graph.mse(y.clone());

        graph.forward();

        let add_output = graph.node(4); 
        let tan_output = graph.node(5);

        assert_eq!(
            add_output.output(),
            arr2(&[[14.0],[20.0],[26.0],[32.0],[38.0]])
        );

        assert_eq!(
            tan_output.output(),
            arr2(&[[0.9999999999986171],[1.0],[1.0],[1.0],[1.0]])
        );

        graph.backward(); 

        assert_eq!(
            graph.node(4).grad(),
            arr2(&[
                [-9.000000000001382],
                [-11.0],
                [-13.0],
                [-15.0],
                [-17.0]
            ])
        );

    }


}

