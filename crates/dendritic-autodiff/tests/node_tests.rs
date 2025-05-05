
#[cfg(test)]
mod node_tests {

    use dendritic_autodiff::tensor::{Tensor};
    use dendritic_autodiff::node::{Node, Node2};
    use dendritic_autodiff::ops::*;
    

    #[test]
    fn test_node_constructors() {

        let mut node_val = Node::val(10.0); 
        assert_eq!(node_val.inputs().len(), 0); 
        assert_eq!(node_val.upstream().len(), 0); 
        assert_eq!(node_val.output(), 10.0);

        let mut binary_node_val: Node<f64> = Node::binary(
            0, 1, 
            Operation::add()
        );

        assert_eq!(binary_node_val.inputs().len(), 2); 
        assert_eq!(binary_node_val.inputs(), vec![0, 1]);
        assert_eq!(binary_node_val.upstream().len(), 0);

    }

    #[test]
    fn test_node_forward() {

        let mut nodes: Vec<Node<f64>> = Vec::new();

        // these go out of scope after adding to node vector
        let a: Node<f64> = Node::val(5.0); 
        let b: Node<f64> = Node::val(10.0);
        let mut add_node = Node::binary(0, 1, Operation::add());

        nodes.push(a); 
        nodes.push(b); 
        nodes.push(add_node.clone()); 

        let a_output = (nodes[0].operation.forward)(&nodes, 0); 
        let b_output = (nodes[1].operation.forward)(&nodes, 1); 
        let output = (nodes[2].operation.forward)(&nodes, 2); 

        assert_eq!(a_output, 5.0); 
        assert_eq!(b_output, 10.0); 
        assert_eq!(output, 15.0); 
    }

    #[test]
    fn test_node_backward() {

        let mut nodes: Vec<Node<f64>> = Vec::new();

        // these go out of scope after adding to node vector
        let a: Node<f64> = Node::val(5.0); 
        let b: Node<f64> = Node::val(10.0);
        let mut add_node = Node::binary(0, 1, Operation::add());
        let inputs_clone = add_node.inputs(); 

        nodes.push(a); 
        nodes.push(b); 
        nodes.push(add_node.clone()); 

        let a_output = (nodes[0].operation.forward)(&nodes, 0); 
        let b_output = (nodes[1].operation.forward)(&nodes, 1); 
        let output = (nodes[2].operation.forward)(&nodes, 2);

        nodes[2].set_output(output); 
        (nodes[2].operation.backward)(&mut nodes, 2);

        assert_eq!(nodes[1].grad(), 1.0); 
        assert_eq!(nodes[0].grad(), 1.0); 
    }


    #[test]
    fn test_node_trait_generic() {


        let a = Node2::val(5.0); 
        let b = Node2::val(10.0);
        let mut add_node: Node2<f64> = Node2::binary(0, 1, Box::new(Add));
        let mut nodes = vec![a, b, add_node];

        let val = nodes[2].forward(&nodes, 2);
        println!("OUTPUT ADD: {:?}", val);

        let call = nodes[2].clone(); 
        call.backward(&mut nodes, 2);

        println!("{:?}", nodes[1]); 
        println!("{:?}", nodes[0]); 

        

    }
}
