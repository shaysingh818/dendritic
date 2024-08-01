
#[cfg(test)]
mod node_tree_tests {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use trees::node::*; 

    #[test]
    fn test_node_empty_children() {


        let features: NDArray<f64> = NDArray::array(
            vec![5, 4],
            vec![
                5.1, 3.5, 1.4, 0.2,
                4.9, 3.0, 1.4, 0.2,
                4.7, 3.2, 1.3, 0.2,
                4.6, 3.1, 1.5, 0.2,
                5.0, 3.6, 1.4, 0.2
            ]
        ).unwrap();
        
        let l1 = Node::leaf(1.0);
        assert_eq!(l1.data().shape().values(), vec![1, 1]);

    }

    #[test]
    fn test_node_creation() {

        /* validating that nodes with empty children can be created */
        let features: NDArray<f64> = NDArray::array(
            vec![5, 4],
            vec![
                5.1, 3.5, 1.4, 0.2,
                4.9, 3.0, 1.4, 0.2,
                4.7, 3.2, 1.3, 0.2,
                4.6, 3.1, 1.5, 0.2,
                5.0, 3.6, 1.4, 0.2
            ]
        ).unwrap();

        let l1 = Node::leaf(1.0);
        let l2 = Node::leaf(2.0);

        let node = Node::new(
            features,
            4.9, 0, 0.0,
            l1,l2
        );

        assert_eq!(node.data().shape().values(), vec![5,4]);
        assert_eq!(node.threshold(), 4.9); 
        assert_eq!(node.feature_idx(), 0); 
        assert_eq!(node.information_gain(), Some(0.0)); 

    }

    #[test]
    fn test_save_node() -> std::io::Result<()> {

        /* validating that nodes with empty children can be created */
        let features: NDArray<f64> = NDArray::array(
            vec![5, 4],
            vec![
                5.1, 3.5, 1.4, 0.2,
                4.9, 3.0, 1.4, 0.2,
                4.7, 3.2, 1.3, 0.2,
                4.6, 3.1, 1.5, 0.2,
                5.0, 3.6, 1.4, 0.2
            ]
        ).unwrap();

        let l1 = Node::leaf(1.0);
        let l2 = Node::leaf(2.0);

        let node = Node::new(
            features,
            4.9, 0, 0.0,
            l1,l2
        );

        let _ = node.save("data/test");

        Ok(())

    }

    #[test]
    fn test_load_node() -> std::io::Result<()> {

        let features: NDArray<f64> = NDArray::array(
            vec![5, 4],
            vec![
                5.1, 3.5, 1.4, 0.2,
                4.9, 3.0, 1.4, 0.2,
                4.7, 3.2, 1.3, 0.2,
                4.6, 3.1, 1.5, 0.2,
                5.0, 3.6, 1.4, 0.2
            ]
        ).unwrap();

        let l1 = Node::leaf(1.0);
        let l2 = Node::leaf(2.0);

        let node = Node::new(
            features,
            4.9, 0, 0.0,
            l1,l2
        );

        let node_save = node.save("test");
        let loaded_node = Node::load(node_save);

        assert_eq!(loaded_node.data().shape().values(), vec![1,1]);
        assert_eq!(loaded_node.threshold(), 4.9); 
        assert_eq!(loaded_node.feature_idx(), 0); 
        assert_eq!(loaded_node.information_gain(), Some(0.0)); 

        Ok(())
    }





}
