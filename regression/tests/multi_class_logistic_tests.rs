
#[cfg(test)]
mod logistic_graph_test {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*;
    use metrics::loss::*;
    use metrics::activations::*;
    use metrics::utils::*; 
    use autodiff::node::{Node, Value}; 
    use autodiff::ops::*;
    use regression::logistic::MultiClassLogistic; 

    #[test]
    fn test_multi_class_logistic() {
        
        let x_path = "data/logistic_modeling_data/multi_class_input";
        let y_path = "data/logistic_modeling_data/multi_class_output";

        let x_train = NDArray::load(x_path).unwrap();
        let y_train = NDArray::load(y_path).unwrap();
        let n = x_train.shape().dim(0);

        let num_classes = 3;
        let learning_rate = 0.1; 
        let inputs = Value::new(&x_train);
        let outputs = Value::new(&y_train);

        let weights_nd = NDArray::new(
            vec![x_train.shape().dim(1), num_classes]
        ).unwrap();
        let bias_nd = NDArray::new(vec![1, 1]).unwrap();

        let mut weights = Value::new(&weights_nd);
        let mut bias = Value::new(&bias_nd); 


        let mut logistic = ScaleAdd::new(
            Dot::new(inputs.clone(), weights.clone()),
            bias.clone()
        ); 


        for epoch in 0..1000 {

            logistic.forward();
            let y_pred = apply(logistic.value(), 0, softmax);

            let loss = categorical_cross_entropy(&y_pred, &y_train).unwrap();
            println!("{:?}", loss); 
            let error = y_pred.subtract(y_train.clone()).unwrap();
            logistic.backward(error.clone());

            /* update weights*/ 
            let learning_rate_factor = 1.0 / n as f64 * learning_rate;

            let w_grad = inputs
                .grad()
                .scalar_mult(learning_rate_factor)
                .unwrap();


            let dw = weights.val().subtract(w_grad).unwrap();
            weights.set_val(&dw); 

            /* update biases */
            let b_collapse = bias
                .grad()
                .sum_axis(1)
                .unwrap();

            let db = b_collapse.scalar_mult(learning_rate_factor).unwrap();
            bias.set_val(&db);

            if epoch == 999 {
                println!("{:?}", y_pred.argmax(0)); 
            }

        }
    }

        
    #[test]
    pub fn test_dot_product_investigate() {

        let a_path = "data/investigation/X_T";
        let b_path = "data/investigation/Y_P";

        let a = NDArray::load(a_path).unwrap();
        let b = NDArray::load(b_path).unwrap();

        let dot_result = a.dot(b).unwrap();
        println!("{:?}", dot_result.values()); 

    }

    #[test]
    pub fn test_multi_logistic_regression() {

        let x_path = "data/logistic_modeling_data/multi_class_input";
        let y_path = "data/logistic_modeling_data/multi_class_output";

        let x_train = NDArray::load(x_path).unwrap();
        let y_train = NDArray::load(y_path).unwrap();

        let mut model = MultiClassLogistic::new(
            x_train.clone(),
            y_train,
            softmax,
            0.1
        ).unwrap();

        model.train(1000, false);
        let results = model.predict(x_train);

        let expected_predictions = vec![
            0.0, 1.0, 2.0, 0.0, 1.0, 
            2.0, 0.0, 1.0, 2.0, 0.0
        ];

        assert_eq!(results.values(), &expected_predictions); 

        

    }




}
