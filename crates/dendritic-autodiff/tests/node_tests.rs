
#[cfg(test)]
mod node_tests {

    use dendritic_autodiff::tensor::{Tensor};
    use dendritic_autodiff::node::{Node, Operation};
    use dendritic_autodiff::ops::{Add, Sub, Mul, Div}; 

    #[test]
    fn test_add_node() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_add_node = Node::binary(
            a, 
            b, 
            Box::new(Add)
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

    #[test]
    fn test_sub_node() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_sub_node = Node::binary(
            a, 
            b, 
            Box::new(Sub)
        );

        assert_eq!(b_sub_node.inputs().len(), 2); 
        assert_eq!(b_sub_node.output().value(), &10.0);

        let lhs_b_sub = &b_sub_node.inputs()[0];
        let rhs_b_sub = &b_sub_node.inputs()[1];

        assert_eq!(lhs_b_sub.value(), &5.0);
        assert_eq!(rhs_b_sub.value(), &10.0);
        assert_eq!(lhs_b_sub.grad(), &5.0);
        assert_eq!(rhs_b_sub.grad(), &10.0);

        b_sub_node.forward();

        assert_eq!(b_sub_node.output().value(), &-5.0); 
    }


    #[test]
    fn test_mul_node() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_mul_node = Node::binary(
            a, 
            b, 
            Box::new(Mul)
        );

        assert_eq!(b_mul_node.inputs().len(), 2); 
        assert_eq!(b_mul_node.output().value(), &10.0);

        let lhs_b_mul = &b_mul_node.inputs()[0];
        let rhs_b_mul = &b_mul_node.inputs()[1];

        assert_eq!(lhs_b_mul.value(), &5.0);
        assert_eq!(rhs_b_mul.value(), &10.0);
        assert_eq!(lhs_b_mul.grad(), &5.0);
        assert_eq!(rhs_b_mul.grad(), &10.0);

        b_mul_node.forward();

        assert_eq!(b_mul_node.output().value(), &50.0); 
    }


    #[test]
    fn test_div_node() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_div_node = Node::binary(
            a, 
            b, 
            Box::new(Div)
        );

        assert_eq!(b_div_node.inputs().len(), 2); 
        assert_eq!(b_div_node.output().value(), &10.0);

        let lhs_b_div = &b_div_node.inputs()[0];
        let rhs_b_div = &b_div_node.inputs()[1];

        assert_eq!(lhs_b_div.value(), &5.0);
        assert_eq!(rhs_b_div.value(), &10.0);
        assert_eq!(lhs_b_div.grad(), &5.0);
        assert_eq!(rhs_b_div.grad(), &10.0);

        b_div_node.forward();

        assert_eq!(b_div_node.output().value(), &0.5); 
    }


    #[test]
    fn test_forward_multiple() {

        // (a+b) -> (a-b)
        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut add_node = Node::binary(
            a.clone(), 
            b.clone(), 
            Box::new(Add)
        );

        let mut sub_node = Node::binary(
            a, 
            b, 
            Box::new(Sub)
        );

        assert_eq!(add_node.inputs().len(), 2); 
        assert_eq!(sub_node.inputs().len(), 2); 

        assert_eq!(add_node.output().value(), &10.0); 
        assert_eq!(sub_node.output().value(), &10.0); 

        let mut items = vec![add_node, sub_node];
        for mut node in &mut items {
            node.forward();
        }

        assert_eq!(items[0].output().value(), &15.0); 
        assert_eq!(items[1].output().value(), &-5.0); 

    }

}
