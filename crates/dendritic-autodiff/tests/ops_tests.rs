
#[cfg(test)]
mod value_test {

    use ndarray::prelude::*; 
    use ndarray::{arr2};
    use dendritic_autodiff::tensor::{Tensor};
    use dendritic_autodiff::graph::{
        Dendrite, 
        Node, 
        BinaryOperation, 
        UnaryOperation
    }; 

    #[test]
    fn test_add_op_scalar() {

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

        (current_node.forward)(&mut torch.nodes[torch.current_node]);
        assert_eq!(torch.nodes[torch.current_node].output.value(), &8.0);

        let new_curr_node = torch.current_node(); 
        let lhs = new_curr_node.inputs()[0].value(); 
        let rhs = new_curr_node.inputs()[1].value();
        let output = new_curr_node.output().value(); 

        assert_eq!(lhs, &5.0); 
        assert_eq!(rhs, &3.0);
        assert_eq!(output, &8.0); 

    }


    #[test]
    fn test_sub_op_scalar() {

        let mut torch = Dendrite::new();
        let expression = torch.sub(5.0, 3.0); 

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

        (current_node.forward)(&mut torch.nodes[torch.current_node]);
        assert_eq!(torch.nodes[torch.current_node].output.value(), &2.0);

        let new_curr_node = torch.current_node(); 
        let lhs = new_curr_node.inputs()[0].value(); 
        let rhs = new_curr_node.inputs()[1].value();
        let output = new_curr_node.output().value(); 

        assert_eq!(lhs, &5.0); 
        assert_eq!(rhs, &3.0);
        assert_eq!(output, &2.0);

        // Order of operations validation
        let mut pemdas = Dendrite::new();
        pemdas.sub(3.0, 5.0);

        let mut curr_node = pemdas.current_node();
        (curr_node.forward)(&mut pemdas.nodes[pemdas.current_node]);
        assert_eq!(pemdas.nodes[pemdas.current_node].output.value(), &-2.0);

    }

}
