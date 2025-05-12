#[cfg(test)]
mod operations_test {

    use dendritic_autodiff::node::{Node};
    use dendritic_autodiff::graph::{ComputationGraph};
    use dendritic_autodiff::graph_interface::*;
    use dendritic_autodiff::ops::*;
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
        nd_graph.add(vec![a, b]);
        nd_graph.add(vec![c]);

        assert_eq!(nd_graph.nodes().len(), 5); 
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

        //println!("{:?}", nd_graph.node(4).grad()); 
        //println!("{:?}", nd_graph.node(3).grad()); 
        //println!("{:?}", nd_graph.node(2).grad()); 


    }


    #[test]
    fn test_subtract() {

        let mut nodes: Vec<Node<f64>> = Vec::new(); 
        let a: Node<f64> = Node::val(2.0); 
        let b: Node<f64> = Node::val(3.0); 
        let add = Node::binary(0, 1, Box::new(Sub)); 

        nodes.push(a);
        nodes.push(b); 
        nodes.push(add);

        let a_val = nodes[0].operation.forward(&nodes, 0); 
        let b_val = nodes[1].operation.forward(&nodes, 1); 
        let sub_val = nodes[2].operation.forward(&nodes, 2); 

        assert_eq!(a_val, 2.0); 
        assert_eq!(b_val, 3.0); 
        assert_eq!(sub_val, -1.0);

        let mut node_2 = nodes[2].clone(); 
        let mut node_1 = nodes[1].clone(); 
        let mut node_0 = nodes[0].clone(); 

        node_2.backward(&mut nodes, 2);  
        node_1.backward(&mut nodes, 1); 
        node_0.backward(&mut nodes, 0);

        assert_eq!(nodes[1].grad(), 1.0); 
        assert_eq!(nodes[0].grad(), 1.0); 

    }


    #[test]
    fn test_multiply() {

        let mut nodes: Vec<Node<f64>> = Vec::new(); 
        let a: Node<f64> = Node::val(2.0); 
        let b: Node<f64> = Node::val(3.0); 
        let add = Node::binary(0, 1, Box::new(Mul)); 

        nodes.push(a);
        nodes.push(b); 
        nodes.push(add);

        let a_val = nodes[0].operation.forward(&nodes, 0); 
        let b_val = nodes[1].operation.forward(&nodes, 1); 
        let mul_val = nodes[2].operation.forward(&nodes, 2); 

        assert_eq!(a_val, 2.0); 
        assert_eq!(b_val, 3.0); 
        assert_eq!(mul_val, 6.0);

        let mut node_2 = nodes[2].clone(); 
        let mut node_1 = nodes[1].clone(); 
        let mut node_0 = nodes[0].clone(); 

        node_2.backward(&mut nodes, 2);  
        node_1.backward(&mut nodes, 1); 
        node_0.backward(&mut nodes, 0);

        assert_eq!(nodes[1].grad(), 2.0); 
        assert_eq!(nodes[0].grad(), 3.0); 

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


}

