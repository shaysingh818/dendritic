
#[cfg(test)]
mod binary_ops_graph_test {

    use ndarray::prelude::*; 
    use ndarray::{arr2};
    use dendritic_autodiff::tensor::Tensor;
    use dendritic_autodiff::graph::Dendrite; 
    use dendritic_autodiff::node::Node; 
    use dendritic_autodiff::binary::*;


    #[test]
    fn test_binary_add() {

        let mut torch = Dendrite::new();
        torch.add(10.0, 13.0); 

        assert_eq!(torch.nodes().len(), 1);
        
        let add_node = torch.node(0);  
        let expected_inputs: Vec<f64> = vec![10.0, 13.0];
        let node_binding = add_node.borrow_mut(); 
        let nodes = node_binding.inputs().iter().enumerate();

        for (index, item) in nodes {
           assert_eq!(expected_inputs[index], item.value());  
        }
        assert!(torch.adj_list.contains_key(&0));
        assert_eq!(torch.current_node_idx(), 1); 
    }

    /*


    #[test]
    fn test_binary_sub() {

        let mut torch = Dendrite::new();
        torch.sub(10.0, 13.0); 

        assert_eq!(torch.nodes().len(), 1);
        
        let sub_node = torch.node(0);  
        let expected_inputs: Vec<f32> = vec![10.0, 13.0];
        let node_binding = sub_node.borrow_mut(); 
        let nodes = node_binding.inputs().iter().enumerate();

        for (index, item) in nodes {
           assert_eq!(expected_inputs[index], item.value());  
        }
        assert!(torch.adj_list.contains_key(&0));
        assert_eq!(torch.current_node_idx(), 1); 

    }

    #[test]
    fn test_binary_mul() {

        let mut torch = Dendrite::new();
        torch.mul(10, 13); 

        assert_eq!(torch.nodes().len(), 1);
        
        let mul_node = torch.node(0); 
        let expected_inputs: Vec<usize> = vec![10, 13];
        let node_binding = mul_node.borrow_mut(); 
        let nodes = node_binding.inputs().iter().enumerate();

        for (index, item) in nodes {
           assert_eq!(expected_inputs[index], item.value());  
        }
        assert!(torch.adj_list.contains_key(&0));
        assert_eq!(torch.current_node_idx(), 1); 

    }

    #[test]
    fn test_binary_div() {

        let mut torch = Dendrite::new();
        torch.div(20, 10); 

        assert_eq!(torch.nodes().len(), 1);
        
        let div_node = torch.node(0); 
        let expected_inputs: Vec<i64> = vec![20, 10];
        let node_binding = div_node.borrow_mut(); 
        let nodes = node_binding.inputs().iter().enumerate();

        for (index, item) in nodes {
           assert_eq!(expected_inputs[index], item.value());  
        }
        assert!(torch.adj_list.contains_key(&0));
        assert_eq!(torch.current_node_idx(), 1); 
    }

    #[test]
    fn test_unrelated_binary_operations() {

        let mut torch = Dendrite::new();

        // create unrelated operations on graph
        torch.add(5.0, 3.0);
        torch.sub(10.0, 3.0); 
        torch.mul(100.0, 3.0); 
        torch.div(100.0, 10.0);

        assert_eq!(torch.nodes().len(), 4);

        let mut expected_outputs = vec![3.0, 3.0, 3.0, 10.0];

        for (index, value) in torch.nodes().iter().enumerate() { 
            assert_eq!(
                expected_outputs[index],
                value.borrow_mut().output().value()
            );
        }

        assert!(torch.adj_list.contains_key(&0));
        assert!(torch.adj_list.contains_key(&1));
        assert!(torch.adj_list.contains_key(&2));
        assert!(torch.adj_list.contains_key(&3));
    } */


}
