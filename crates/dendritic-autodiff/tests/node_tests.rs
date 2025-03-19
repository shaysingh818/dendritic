
#[cfg(test)]
mod node_tests {

    use dendritic_autodiff::tensor::{Tensor};
    use dendritic_autodiff::graph::{
        Node, 
        Operation, 
        Add as AddOp, 
        Sub as SubOp
    }; 

    #[test]
    fn test_binary_node() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_add_node = Node::binary(
            a.clone(), 
            b.clone(), 
            AddOp::forward, 
            AddOp::backward
        );

        assert_eq!(b_add_node.inputs().len(), 2); 
        assert_eq!(b_add_node.output().value(), &10.0);

        let lhs_b_add = &b_add_node.inputs()[0];
        let rhs_b_add = &b_add_node.inputs()[1];

        assert_eq!(lhs_b_add.value(), &5.0);
        assert_eq!(rhs_b_add.value(), &10.0);
        assert_eq!(lhs_b_add.grad(), &5.0);
        assert_eq!(rhs_b_add.grad(), &10.0);

        b_add_node.forward();

        assert_eq!(b_add_node.output().value(), &15.0);
    }

}
