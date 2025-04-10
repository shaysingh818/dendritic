
#[cfg(test)]
mod node_tests {

    use dendritic_autodiff::tensor::{Tensor};
    use dendritic_autodiff::node::{Node};
    //use dendritic_autodiff::ops::{Add, Sub, Mul, Div};

    /*
    #[test]
    fn test_node_forward() {
 
        let mut b_add_node = Node::binary(
            5.0, 
            10.0, 
            Box::new(Add)
        );

        b_add_node.forward(None);

        assert_eq!(b_add_node.inputs().len(), 2);
        assert_eq!(b_add_node.output().value(), 15.0);

        let mut u_add_node = Node::unary(10.0, Box::new(Add));

        assert_eq!(u_add_node.inputs().len(), 1);
        assert_eq!(u_add_node.output().value(), 10.0); 

        u_add_node.forward(Some(&b_add_node));

        assert_eq!(u_add_node.inputs().len(), 1);
        assert_eq!(u_add_node.output().value(), 25.0); 
    }

    #[test]
    fn test_generic_trait_impl() {

        let a: f64 = 5.0; 
        let b: i32 = 10;
        let c: i64 = 10;
        let d: u8 = 10;
        let e: u16 = 100;

        // validating trait macro impl for scalar numeric values 
        let mut f64_node = Node::binary(a.clone(), a, Box::new(Add));
        let mut i32_node = Node::binary(b.clone(), b, Box::new(Sub));
        let mut i64_node = Node::binary(c.clone(), c, Box::new(Mul));
        let mut u8_node = Node::binary(d.clone(), d, Box::new(Div));
        let mut u16_node = Node::binary(e.clone(), e, Box::new(Add));

        f64_node.forward(None); 
        i32_node.forward(None); 
        i64_node.forward(None); 
        u8_node.forward(None); 
        u16_node.forward(None);

        assert_eq!(f64_node.output().value(), 10.0); 
        assert_eq!(i32_node.output().value(), 0); 
        assert_eq!(i64_node.output().value(), 100); 
        assert_eq!(u8_node.output().value(), 1); 
        assert_eq!(u16_node.output().value(), 200); 

        /*
        f64_node.backward(None, f64_node); 
        i32_node.backward(None, i32_node); 
        i64_node.backward(None); 
        u8_node.backward(None); 
        u16_node.backward(None); */ 

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

        assert_eq!(add_node.output().value(), 10.0); 
        assert_eq!(sub_node.output().value(), 10.0); 

        let mut items = vec![add_node, sub_node];
        for node in &mut items {
            node.forward(None);
        }

        assert_eq!(items[0].output().value(), 15.0); 
        assert_eq!(items[1].output().value(), -5.0); 

    }  */



}
