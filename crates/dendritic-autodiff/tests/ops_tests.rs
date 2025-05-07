#[cfg(test)]
mod operations_test {

    use dendritic_autodiff::node::{Node};
    use dendritic_autodiff::graph::{ComputationGraph};
    use dendritic_autodiff::ops::*;
    use ndarray::prelude::*; 
    use ndarray::{arr2};

    #[test]
    fn test_scalar_generic_add() {

        let mut nodes: Vec<Node<f64>> = Vec::new(); 
        let a: Node<f64> = Node::val(2.0); 
        let b: Node<f64> = Node::val(3.0); 
        let add = Node::binary(0, 1, Box::new(Add)); 

        nodes.push(a);
        nodes.push(b); 
        nodes.push(add);

        let a_val = nodes[0].operation.forward(&nodes, 0); 
        let b_val = nodes[1].operation.forward(&nodes, 1); 
        let add_val = nodes[2].operation.forward(&nodes, 2); 

        assert_eq!(a_val, 2.0); 
        assert_eq!(b_val, 3.0); 
        assert_eq!(add_val, 5.0);

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
    fn test_scalar_generic_sub() {

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
    fn test_scalar_generic_mul() {

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


}

