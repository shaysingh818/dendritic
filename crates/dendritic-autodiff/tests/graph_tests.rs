
#[cfg(test)]
mod graph_test {

    use dendritic_autodiff::graph::{Dendrite};
    use dendritic_autodiff::ops::{Add};
    use dendritic_autodiff::error::{GraphError};
    use dendritic_autodiff::unary::*; 
    use dendritic_autodiff::binary::*; 
    use ndarray::prelude::*; 
    use ndarray::{arr2};

    /*

    #[test]
    fn test_graph_instantiation() {

        let graph: Dendrite<f64> = Dendrite::new();

        assert_eq!(graph.nodes().len(), 0); 
        assert_eq!(graph.current_node_idx(), 0);
        assert_eq!(graph.prev_node_idx(), 0);
        assert_eq!(graph.path().len(), 0);

    }

    #[test]
    fn test_graph_node_borrow() {
 
        let mut graph: Dendrite<f64> = Dendrite::new();

        assert_eq!(graph.nodes().len(), 0); 
        assert_eq!(graph.current_node_idx(), 0);

        graph.binary(10.0, 10.0, Box::new(Add)); 
        graph.binary(10.0, 100.0, Box::new(Add));

        let mut node_1 = graph.node(0);
        let mut node_2 = graph.node(1);

        node_1.borrow_mut().forward(None);
        node_2.borrow_mut().forward(
            Some(&node_1.borrow_mut())
        );

        assert_eq!(node_1.borrow_mut().output().value(), 20.0);
        assert_eq!(node_2.borrow_mut().output().value(), 110.0);

        node_1.borrow_mut().set_input(0, 50.0);
        node_2.borrow_mut().set_input(0, 30.0);

        let mut node_1_ref = graph.node(0);
        let mut node_2_ref = graph.node(1);

        assert_eq!(node_1_ref.borrow_mut().inputs()[0].value(), 50.0); 
        assert_eq!(node_2_ref.borrow_mut().inputs()[0].value(), 30.0); 

        node_1_ref.borrow_mut().forward(None);
        node_2_ref.borrow_mut().forward(Some(&node_1_ref.borrow_mut()));
    
        assert_eq!(node_1_ref.borrow_mut().output().value(), 60.0);  
        assert_eq!(node_2_ref.borrow_mut().output().value(), 130.0); 
    }

    #[test]
    fn test_graph_binary_node() {

        let mut graph: Dendrite<f64> = Dendrite::new();

        assert_eq!(graph.nodes().len(), 0); 
        assert_eq!(graph.current_node_idx(), 0);

        graph.binary(3.0, 4.0, Box::new(Add)); 

        assert_eq!(graph.nodes().len(), 1);
        assert_eq!(graph.current_node_idx(), 1); 
        assert_eq!(graph.prev_node_idx(), 0);
        assert_eq!(graph.next_node_idx(), 0);
        assert!(graph.adj_list.contains_key(&0));

        let neighbors = graph.adj_list.get(&0).unwrap(); 
        assert_eq!(neighbors.len(), 0); 

    }

    #[test]
    fn test_graph_unary_node() -> Result<(), Box<dyn std::error::Error>>  {

        let mut graph: Dendrite<f64> = Dendrite::new();

        assert_eq!(graph.nodes().len(), 0); 
        assert_eq!(graph.current_node_idx(), 0);

        graph.binary(3.0, 4.0, Box::new(Add));
        graph.unary(100.0, Box::new(Add)); 

        assert_eq!(graph.nodes().len(), 2);
        assert_eq!(graph.current_node_idx(), 2); 
        assert_eq!(graph.prev_node_idx(), 1);
        assert_eq!(graph.next_node_idx(), 0);
        assert_eq!(graph.adj_list.get(&0).unwrap().len(), 1);
        assert_eq!(graph.adj_list.get(&1).unwrap().len(), 0);

        let mut torch: Dendrite<f64> = Dendrite::new(); 

        let value = torch.unary(100.0, Box::new(Add)).unwrap_err(); 
        matches!(value, GraphError::UnaryOperation); 
        Ok(())
    }

    #[test]
    fn test_graph_node_relationships() -> Result<(), Box<dyn std::error::Error>> {

        let b: f64 = 1.0;
        let w: f64 = 10.0;
        let x: f64 = 20.0; 

        let mut graph: Dendrite<f64> = Dendrite::new();

        graph.mul(x, w); 
        graph.u_add(b); 
        graph.u_mul(w.clone()); 
        graph.u_add(b.clone());

        // validate nodes created in graph
        let expected_keys: Vec<usize> = vec![0, 1, 2, 3];
        for key in expected_keys {
            assert!(graph.adj_list.contains_key(&key));
        } 

        // validate relationships in graph
        let mul = graph.adj_list.get(&0).unwrap();
        let u_add = graph.adj_list.get(&1).unwrap();
        let u_mul = graph.adj_list.get(&2).unwrap();
        let u_add_2 = graph.adj_list.get(&3).unwrap();


        assert_eq!(mul.get(&1).unwrap(), &1);
        assert_eq!(u_add.get(&2).unwrap(), &2);
        assert_eq!(u_mul.get(&3).unwrap(), &3);
        assert_eq!(u_mul.get(&4), None);

        Ok(())
    }


    #[test]
    fn test_graph_forward() -> Result<(), Box<dyn std::error::Error>>  {

        let b: f64 = 1.0;
        let w: f64 = 10.0;
        let x: f64 = 20.0; 

        let mut graph: Dendrite<f64> = Dendrite::new();

        graph.mul(x, w); 
        graph.u_add(b); 
        graph.u_mul(w.clone()); 
        graph.u_add(b.clone());

        let node_0 = graph.node(0); 
        let node_1 = graph.node(1); 
        let node_2 = graph.node(2); 
        let node_3 = graph.node(3);

        assert_eq!(node_0.borrow_mut().output().value(), 10.0); 
        assert_eq!(node_1.borrow_mut().output().value(), 1.0);
        assert_eq!(node_2.borrow_mut().output().value(), 10.0); 
        assert_eq!(node_3.borrow_mut().output().value(), 1.0);
        assert_eq!(graph.path(), vec![]); 

        graph.forward(); 

        let mul = graph.node(0); 
        let u_add = graph.node(1); 
        let u_mul = graph.node(2); 
        let u_add_2 = graph.node(3);

        assert_eq!(mul.borrow_mut().output().value(), 200.0); 
        assert_eq!(u_add.borrow_mut().output().value(), 201.0); 
        assert_eq!(u_mul.borrow_mut().output().value(), 2010.0); 
        assert_eq!(u_add_2.borrow_mut().output().value(), 2011.0); 
        assert_eq!(graph.path(), vec![0, 1, 2, 3]); 

        Ok(())

    }

    #[test]
    fn test_graph_backward() -> Result<(), Box<dyn std::error::Error>>  {
        
        let b: f64 = 1.0;
        let w: f64 = 10.0;
        let x: f64 = 20.0; 

        let mut graph: Dendrite<f64> = Dendrite::new();

        graph.mul(x, w); 
        graph.u_add(b); 
        graph.u_mul(w.clone()); 
        graph.u_add(b.clone());

        graph.forward();



        Ok(())
    } */

}
