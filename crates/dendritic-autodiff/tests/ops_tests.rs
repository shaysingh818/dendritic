#[cfg(test)]
mod operations_test {

    use dendritic_autodiff::node::{Node, Operation};
    use dendritic_autodiff::ops::{Add, Sub, Div, Mul};
    use ndarray::prelude::*; 
    use ndarray::{arr2};

    /*
    #[test]
    fn test_add_op() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_add_node = Node::binary(
            a, 
            b, 
            Box::new(Add)
        );

        assert_eq!(b_add_node.inputs().len(), 2); 
        assert_eq!(b_add_node.output().value(), 10.0);

        let lhs_b_add = &b_add_node.inputs()[0];
        let rhs_b_add = &b_add_node.inputs()[1];

        assert_eq!(lhs_b_add.value(), 5.0);
        assert_eq!(rhs_b_add.value(), 10.0);
        assert_eq!(lhs_b_add.grad(), 5.0);
        assert_eq!(rhs_b_add.grad(), 10.0);

        b_add_node.forward(None);

        assert_eq!(b_add_node.output().value(), 15.0);

        /*
        b_add_node.backward(None);

        let lhs_grad = b_add_node.inputs()[0].grad();
        let rhs_grad = b_add_node.inputs()[1].grad();

        assert_eq!(lhs_grad, 1.0);
        assert_eq!(rhs_grad, 1.0);

        let mut u_add_node = Node::unary(10.0, Box::new(Add));

        assert_eq!(u_add_node.inputs().len(), 1);
        assert_eq!(u_add_node.output().value(), 10.0); 

        u_add_node.forward(Some(&b_add_node));

        assert_eq!(u_add_node.inputs().len(), 1);
        assert_eq!(u_add_node.output().value(), 25.0);

        // Using ndarrays (non scalar values)
        let a = arr2(&[[0.0, 0.0, 1.0], [1.0, 2.0, 2.0]]); 
        let b = arr2(&[[1.0, 1.0, 1.0]]);
        let c = arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]); 

        let mut b_add_arrays = Node::binary(a, b, Box::new(Add));

        b_add_arrays.forward(None);

        let output = b_add_arrays.output().value(); 
        assert_eq!(output.shape(), vec![2, 3]);
        assert_eq!(output, arr2(&[[1.0, 1.0, 2.0], [2.0, 3.0, 3.0]]));
        
        let mut u_add_arrays = Node::unary(c, Box::new(Add)); 

        u_add_arrays.forward(Some(&b_add_arrays));

        let u_output = u_add_arrays.output().value();
        assert_eq!(u_output.shape(), vec![2, 3]);
        assert_eq!(u_output, arr2(&[[2.0, 2.0, 3.0], [3.0, 4.0, 4.0]])); */ 
    }

    #[test]
    fn test_sub_op() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_sub_node = Node::binary(
            a, 
            b, 
            Box::new(Sub)
        );

        assert_eq!(b_sub_node.inputs().len(), 2); 
        assert_eq!(b_sub_node.output().value(), 10.0);

        let lhs_b_sub = &b_sub_node.inputs()[0];
        let rhs_b_sub = &b_sub_node.inputs()[1];

        assert_eq!(lhs_b_sub.value(), 5.0);
        assert_eq!(rhs_b_sub.value(), 10.0);
        assert_eq!(lhs_b_sub.grad(), 5.0);
        assert_eq!(rhs_b_sub.grad(), 10.0);

        b_sub_node.forward(None);

        assert_eq!(b_sub_node.output().value(), -5.0);

        /*
        b_sub_node.backward(None); 

        let lhs_grad = b_sub_node.inputs()[0].grad();
        let rhs_grad = b_sub_node.inputs()[1].grad();
        
        assert_eq!(lhs_grad, 1.0);
        assert_eq!(rhs_grad, 1.0); */ 
    }


    #[test]
    fn test_mul_op() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_mul_node = Node::binary(
            a, 
            b, 
            Box::new(Mul)
        );

        assert_eq!(b_mul_node.inputs().len(), 2); 
        assert_eq!(b_mul_node.output().value(), 10.0);

        let lhs_b_mul = &b_mul_node.inputs()[0];
        let rhs_b_mul = &b_mul_node.inputs()[1];

        assert_eq!(lhs_b_mul.value(), 5.0);
        assert_eq!(rhs_b_mul.value(), 10.0);
        assert_eq!(lhs_b_mul.grad(), 5.0);
        assert_eq!(rhs_b_mul.grad(), 10.0);

        b_mul_node.forward(None);

        assert_eq!(b_mul_node.output().value(), 50.0);

        /*
        b_mul_node.backward(None);

        let lhs_grad = b_mul_node.inputs()[0].grad();
        let rhs_grad = b_mul_node.inputs()[1].grad();

        assert_eq!(lhs_grad, 10.0);
        assert_eq!(rhs_grad, 5.0); 

        // testing ndarray mul operation
        let a = arr2(&[
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 3.0],
            [0.0, 0.0, 4.0]
        ]);

        let b = arr2(&[
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);

        let expected_output = arr2(&[
            [3.0, 3.0, 3.0],
            [8.0, 8.0, 8.0],
            [12.0, 12.0, 12.0],
            [12.0, 12.0, 12.0]
        ]);

        let mut b_mul_array = Node::binary(
            a, 
            b, 
            Box::new(Mul)
        );

        b_mul_array.forward(None);

        let output = b_mul_array.output().value();
        assert_eq!(output.shape(), vec![4, 3]); 
        assert_eq!(output, expected_output);

        b_mul_array.backward(Some(output)); */


    }


    #[test]
    fn test_div_op() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0; 

        let mut b_div_node = Node::binary(
            a, 
            b, 
            Box::new(Div)
        );

        assert_eq!(b_div_node.inputs().len(), 2); 
        assert_eq!(b_div_node.output().value(), 10.0);

        let lhs_b_div = &b_div_node.inputs()[0];
        let rhs_b_div = &b_div_node.inputs()[1];

        assert_eq!(lhs_b_div.value(), 5.0);
        assert_eq!(rhs_b_div.value(), 10.0);
        assert_eq!(lhs_b_div.grad(), 5.0);
        assert_eq!(rhs_b_div.grad(), 10.0);

        b_div_node.forward(None);

        assert_eq!(b_div_node.output().value(), 0.5);

        /*
        b_div_node.backward(None);

        let lhs_grad = b_div_node.inputs()[0].grad();
        let rhs_grad = b_div_node.inputs()[1].grad();

        assert_eq!(lhs_grad, 0.1);
        assert_eq!(rhs_grad, 0.05); */ 

    } */

}

