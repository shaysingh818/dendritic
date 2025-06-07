
#[cfg(test)]
mod graph_test {

    use std::any::type_name; 
    use dendritic_autodiff::tensor::Tensor; 
    use dendritic_autodiff::node::Node; 
    use dendritic_autodiff::error::{GraphError};
    use dendritic_autodiff::operations::activation::*; 
    use dendritic_autodiff::operations::arithmetic::*; 
    use dendritic_autodiff::operations::loss::*; 
    use dendritic_autodiff::graph::*; 
    use ndarray::prelude::*; 
    use ndarray::{arr2};


    #[test]
    fn test_graph_instantiation() {

        let graph: ComputationGraph<f64> = ComputationGraph::new();

        assert_eq!(graph.nodes().len(), 0); 
        assert_eq!(graph.curr_node_idx(), -1);
        assert_eq!(graph.path().len(), 0);
        assert_eq!(graph.variables().len(), 0); 
        assert_eq!(graph.operations().len(), 0); 
    }

    #[test]
    fn test_graph_binary_node() {

        let a = Some(5.0); 
        let b = Some(10.0); 

        let mut graph = ComputationGraph::new(); 
        graph.binary(a, b, Box::new(Add));

        assert_eq!(graph.nodes().len(), 3); 
        assert_eq!(graph.curr_node_idx(), 2);

        let a_val = graph.node(0); 
        let b_val = graph.node(1); 
        let add_node = graph.node(2);

        assert_eq!(a_val.upstream(), vec![2]); 
        assert_eq!(b_val.upstream(), vec![2]); 
        assert_eq!(a_val.inputs().len(), 0); 
        assert_eq!(b_val.inputs().len(), 0);

        assert_eq!(a_val.output(), 5.0); 
        assert_eq!(b_val.output(), 10.0);

        assert_eq!(add_node.upstream().len(), 0); 
        assert_eq!(add_node.inputs().len(), 2); 
        assert_eq!(add_node.inputs(), vec![0, 1]);
        assert_eq!(add_node.output(), 0.0); 
    }

    #[test]
    fn test_graph_unary_node() -> Result<(), Box<dyn std::error::Error>>  {

        let a = Some(5.0); 
        let b = Some(10.0);
        let c = 100.0; 

        let mut graph = ComputationGraph::new(); 
        graph.binary(a, b, Box::new(Add));
        graph.unary(c, Box::new(Add)); 

        assert_eq!(graph.nodes().len(), 5); 
        assert_eq!(graph.curr_node_idx(), 4); 

        let a_val = graph.node(0); 
        let b_val = graph.node(1); 
        let add = graph.node(2); 
        let c_val = graph.node(3); 
        let add_2 = graph.node(4);

        assert_eq!(a_val.upstream(), vec![2]); 
        assert_eq!(b_val.upstream(), vec![2]); 
        assert_eq!(a_val.inputs().len(), 0); 
        assert_eq!(b_val.inputs().len(), 0);
        assert_eq!(a_val.output(), 5.0); 
        assert_eq!(b_val.output(), 10.0);

        assert_eq!(add.upstream().len(), 1);
        assert_eq!(add.upstream(), vec![4]); 
        assert_eq!(add.inputs().len(), 2); 
        assert_eq!(add.inputs(), vec![0, 1]);
        assert_eq!(add.output(), 0.0);

        assert_eq!(c_val.inputs().len(), 0); 
        assert_eq!(c_val.upstream().len(), 1); 
        assert_eq!(c_val.upstream(), vec![4]); 
        assert_eq!(c_val.output(), 100.0);

        assert_eq!(add_2.inputs().len(), 2); 
        assert_eq!(add_2.inputs(), vec![2, 3]); 
        assert_eq!(add_2.upstream().len(), 0); 
        assert_eq!(add_2.output(), 0.0); 

        Ok(())
    }

    /*
    #[test]
    fn test_graph_operation_relationships() {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]); 

        assert_eq!(graph.nodes().len(), 9);
        assert_eq!(graph.path().len(), 0);
        assert_eq!(graph.curr_node_idx(), 8);

        let val1 = graph.node(0);
        assert_eq!(val1.upstream(), vec![2]); 
        assert_eq!(val1.inputs().len(), 0); 

        let val2 = graph.node(1);
        assert_eq!(val2.upstream(), vec![2]); 
        assert_eq!(val2.inputs().len(), 0); 

        let add_node = graph.node(2);
        assert_eq!(add_node.upstream(), vec![4]); 
        assert_eq!(add_node.inputs().len(), 2); 
        assert_eq!(add_node.inputs(), vec![0, 1]); 

        let val3 = graph.node(3);
        assert_eq!(val3.upstream(), vec![4]); 
        assert_eq!(val3.inputs().len(), 0); 

        let u_add_node = graph.node(4);
        assert_eq!(u_add_node.upstream(), vec![6]); 
        assert_eq!(u_add_node.inputs().len(), 2);
        assert_eq!(u_add_node.inputs(), vec![2, 3]);

        let val4 = graph.node(5);
        assert_eq!(val4.upstream(), vec![6]); 
        assert_eq!(val4.inputs().len(), 0);

        let u_mul_node = graph.node(6);
        assert_eq!(u_mul_node.upstream(), vec![8]); 
        assert_eq!(u_mul_node.inputs().len(), 2);
        assert_eq!(u_mul_node.inputs(), vec![4, 5]);
        
        let val5 = graph.node(7);
        assert_eq!(val5.upstream(), vec![8]); 
        assert_eq!(val5.inputs().len(), 0);

        let u_sub_node = graph.node(8);
        assert_eq!(u_sub_node.upstream().len(), 0); 
        assert_eq!(u_sub_node.inputs().len(), 2);
        assert_eq!(u_sub_node.inputs(), vec![6, 7]);

    }

    #[test]
    fn test_graph_forward_evaluate_scalar() {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]); 

        graph.forward(); 

        assert_eq!(graph.path().len(), 4);
        assert_eq!(
            graph.path(),
            vec![2, 4, 6, 8]
        );

        let expected_outputs = vec![15.0, 115.0, 2300.0, 2290.0];

        for (idx, node) in graph.path().iter().enumerate() {
            let node_output = graph.node(*node);
            assert_eq!(node_output.output(), expected_outputs[idx]); 
        }
    }

    #[test]
    fn test_graph_backward_evaluate_scalar() {

        let mut graph = ComputationGraph::new();
        graph.add(vec![5.0, 10.0]); 
        graph.add(vec![100.0]);
        graph.mul(vec![20.0]);
        graph.sub(vec![10.0]); 

        graph.forward(); 

        let output_node = graph.curr_node(); 

        graph.backward();

        assert_eq!(graph.path().len(), 4);
        assert_eq!(
            graph.path(),
            vec![2, 4, 6, 8]
        );

        let mut path = graph.path().clone(); 
        path.reverse();

        let vars = graph.variables(); 
        let ops = graph.operations();

        let expected_var_grads = vec![1.0, 1.0, 1.0, 115.0, 1.0];
        let expected_op_grads = vec![1.0, 20.0, 1.0, 0.0];

        for (idx, var) in vars.iter().enumerate() {
            let node = graph.node(*var); 
            assert_eq!(node.grad(), expected_var_grads[idx]); 
        }

        for (idx, op) in ops.iter().enumerate() {
            let node = graph.node(*op); 
            assert_eq!(node.grad(), expected_op_grads[idx]); 
        }

    } */



}
