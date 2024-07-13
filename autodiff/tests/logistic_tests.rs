
#[cfg(test)]
mod logistic_graph_test {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use metrics::loss::*;

    use autodiff::node::*; 
    use autodiff::ops::*;
    use autodiff::regularizers::*;
    use regression::logistic::*; 


    #[test]
    fn test_multi_class_logistic() {
        
        let x: NDArray<f64> = NDArray::array(
            vec![10, 2],
            vec![
                1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 6.0, 7.5, 1.0, 0.6, 9.0, 
                11.0, 8.5, 10.5, 9.2, 11.2, 6.5, 8.0, 1.3, 0.5
            ]
        ).unwrap(); 


        let y: NDArray<f64> = NDArray::array(
            vec![10, 1],
            vec![
                0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0, 0.0, 
            ]
        ).unwrap(); 
        
        let mut log_model = Logistic::new(
            x,
            y,
            softmax,
            0.001
        ).unwrap();

        log_model.sgd(1000, true, 3);



    }

}
