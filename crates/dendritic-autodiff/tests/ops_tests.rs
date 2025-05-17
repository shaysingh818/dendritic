#[cfg(test)]
mod operations_test {

    use dendritic_autodiff::node::{Node};
    use dendritic_autodiff::graph::{ComputationGraph};
    use dendritic_autodiff::graph_interface::*;
    use dendritic_autodiff::ops::*;
    use ndarray::prelude::*; 
    use ndarray::{arr2};

    /*
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
            assert_eq!(
                nd_graph.node(*var).grad(),
                expected_grads[idx]
            );
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

    } */


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

        /*
        let vars = nd_graph.variables();

        for (idx, var) in vars.iter().enumerate() {
            println!("{:?}", nd_graph.node(*var).grad()); 
            /*
            assert_eq!(
                nd_graph.node(*var).grad(),
                expected_grads[idx]
            );*/
        } 

        let vars = nd_graph.variables();
        let expected_grads = vec![a, b, c];

        for (idx, var) in vars.iter().enumerate() {
            /*
            assert_eq!(
                nd_graph.node(*var).grad(),
                expected_grads[idx]
            );*/
        } */


    }


    /*
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

    } */


}

