
#[cfg(test)]
mod value_test {

    use ndarray::prelude::*; 
    use ndarray::{arr2};
    use dendritic_autodiff::tensor::Tensor;
    use dendritic_autodiff::graph::Dendrite; 
    use dendritic_autodiff::node::Node; 
    use dendritic_autodiff::binary::*; 

    #[test]
    fn test_binary_operation() {

        let mut torch = Dendrite::new();
        torch.add(5.0, 3.0);

        assert_eq!(torch.nodes().len(), 1);
        assert_eq!(torch.reference_count(), 0);
        assert_eq!(torch.current_node_idx(), 0);

        let mut current_node = torch.current_node();
        assert_eq!(current_node.inputs.len(), 2);

        let lhs = current_node.inputs()[0].value(); 
        let rhs = current_node.inputs()[1].value();
        let output = current_node.output().value(); 

        assert_eq!(lhs, &5.0); 
        assert_eq!(rhs, &3.0);
        assert_eq!(output, &3.0);

        torch.forward(); 

        let new_curr_node = torch.current_node(); 
        let lhs = new_curr_node.inputs()[0].value(); 
        let rhs = new_curr_node.inputs()[1].value();
        let output = new_curr_node.output().value();

        assert_eq!(lhs, &5.0); 
        assert_eq!(rhs, &3.0);
        assert_eq!(output, &8.0); 

        // combined operations test
        let mut graph = Dendrite::new(); 
        graph.add(5.0, 3.0); 
        graph.sub(10.0, 11.0);

        assert_eq!(graph.nodes().len(), 2);
        assert_eq!(graph.reference_count(), 0);
        assert_eq!(graph.current_node_idx(), 0);

        let add_node = &graph.nodes[0];
        let sub_node = &graph.nodes[1];

        assert_eq!(add_node.inputs().len(), 2); 
        assert_eq!(sub_node.inputs().len(), 2);

        assert_eq!(add_node.output().value(), &3.0); 
        assert_eq!(sub_node.output().value(), &11.0); 



    }


}
