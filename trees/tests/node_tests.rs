
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
            4.9, 0,
            l1,l2
        );

        assert_eq!(node.data().shape().values(), vec![5,4]);

    }

}
