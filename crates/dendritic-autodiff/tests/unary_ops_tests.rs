
#[cfg(test)]
mod unary_ops_test {

    use ndarray::prelude::*; 
    use ndarray::{arr2};
    use dendritic_autodiff::tensor::Tensor;
    use dendritic_autodiff::graph::Dendrite; 
    use dendritic_autodiff::node::Node; 
    use dendritic_autodiff::unary::*;
    use dendritic_autodiff::binary::*;
    use dendritic_autodiff::ops::{Add};

    /*
    #[test]
    fn test_unary_operation_node() {

        let a: f64 = 5.0; 
        let b: f64 = 10.0;

        let mut torch = Dendrite::new(); 
        torch.add(5.0, 10.0); 
        torch.u_add(5.0);

        assert_eq!(torch.nodes().len(), 2); 

        let node_1 = torch.node(0);
        let node_2 = torch.node(1);

        assert_eq!(node_1.borrow_mut().inputs().len(), 2); 
        assert_eq!(node_2.borrow_mut().inputs().len(), 1);

        assert!(torch.adj_list.contains_key(&0)); 
        assert!(torch.adj_list.contains_key(&1));

        let add = torch.adj_list.get(&0).unwrap(); 
        let u_add = torch.adj_list.get(&1).unwrap(); 

        assert_eq!(add.get(&1).unwrap(), &1); 
        assert_eq!(u_add.get(&1), None);

        /*
        let node_0_ref = &mut torch.nodes[0];

        node_0_ref.forward(None);

        let node_1_ref = &mut torch.nodes[1]; 

        node_1_ref.forward(Some(node_0_ref)); */ 

    } */

}
