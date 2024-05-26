use steepgrad::loss; 
use steepgrad::ndarray;
use steepgrad::autodiff;

#[cfg(test)]
mod autodiff_test {

    use crate::ndarray::ndarray::NDArray;
    use crate::ndarray::ops::*;
    use crate::autodiff::node::*; 
    use crate::autodiff::ops::*; 
    use crate::loss::mse::*;


    #[test]
    fn test_value_instance() {

        let y: NDArray<f64> = NDArray::load("data/linear_testing_data/outputs").unwrap();
        let x: NDArray<f64> = NDArray::load("data/linear_testing_data/inputs").unwrap(); 
        let mut x_value = Value::new(&x); 

        let expected_shape = vec![5, 3]; 
        assert_eq!(x_value.val().shape(), &expected_shape); 
        assert_eq!(x_value.val().size(), 15);
        assert_eq!(x_value.val().rank(), 2); 

        assert_eq!(x_value.grad().shape(), &expected_shape); 
        assert_eq!(x_value.grad().size(), 15);
        assert_eq!(x_value.grad().rank(), 2); 
        x_value.set_val(&y);  

        let new_val_shape = vec![5, 1]; 
        assert_eq!(x_value.val().shape(), &new_val_shape); 
        assert_eq!(x_value.val().size(), 5);
        assert_eq!(x_value.val().rank(), 2);
        x_value.set_grad(&y);

        assert_eq!(x_value.grad().shape(), &new_val_shape); 
        assert_eq!(x_value.grad().size(), 5);
        assert_eq!(x_value.grad().rank(), 2);

    }


    #[test]
    fn test_dot_node() {

        let x: NDArray<f64> = NDArray::load("data/linear_testing_data/inputs").unwrap(); 
        let w: NDArray<f64> = NDArray::load("data/linear_testing_data/weights").unwrap();
        let y: NDArray<f64> = NDArray::load("data/linear_testing_data/outputs").unwrap();
        let x_value = Value::new(&x); 
        let w_value = Value::new(&w); 

        let mut dot_op = Dot::new(x_value.clone(), w_value.clone());
        let expected_output_shape = vec![5, 1];

        assert_eq!(dot_op.value().shape(), &expected_output_shape);
        assert_eq!(dot_op.value().rank(), 2);
        assert_eq!(dot_op.value().size(), 5);
        dot_op.forward();

        for item in dot_op.value().values() {
            let val: f64 = 0.0;
            assert_eq!(item, &val); 
        }

        let y_pred = dot_op.value();
        let output = y_pred.subtract(y).unwrap();
        dot_op.backward(output.clone());

        let expected_ws = vec![-230.0, -300.0, -370.0];
        assert_eq!(x_value.grad().shape(), w.shape()); 
        assert_eq!(x_value.grad().rank(), 2); 
        assert_eq!(x_value.grad().values(), &expected_ws);

        assert_eq!(w_value.grad().rank(), 2);
        for item in w_value.grad().values() {
            let val: f64 = 0.0;
            assert_eq!(item, &val); 
        }
        
    }

    #[test]
    fn test_scale_add_node() {

        let y: NDArray<f64> = NDArray::load("data/linear_testing_data/outputs").unwrap();
        let b: NDArray<f64> = NDArray::load("data/linear_testing_data/bias").unwrap();
        let y_value = Value::new(&y); 
        let b_value = Value::new(&b); 

        let mut scale_op = ScaleAdd::new(y_value.clone(), b_value.clone());
        let expected_output_shape = vec![5, 1];
        let expected_vals = vec![11.0, 13.0, 15.0, 17.0, 19.0];
        scale_op.forward(); 

        assert_eq!(scale_op.value().shape(), &expected_output_shape);
        assert_eq!(scale_op.value().rank(), 2);
        assert_eq!(scale_op.value().size(), 5);
        assert_eq!(scale_op.value().values(), &expected_vals);

        let output = y.subtract(scale_op.value()).unwrap();
        scale_op.backward(output);

        let expected_b_grad = vec![-1.0, -1.0, -1.0, -1.0, -1.0];
        assert_eq!(b_value.grad().values(), &expected_b_grad);
        assert_eq!(b_value.grad().shape(), y.shape());
        assert_eq!(b_value.grad().rank(), y.rank());

    }

    #[test]
    fn test_linear_node_mut() {

        let x: NDArray<f64> = NDArray::load("data/linear_testing_data/inputs").unwrap();
        let y: NDArray<f64> = NDArray::load("data/linear_testing_data/outputs").unwrap();
        let w: NDArray<f64> = NDArray::load("data/linear_testing_data/weights").unwrap();
        let b: NDArray<f64> = NDArray::load("data/linear_testing_data/bias").unwrap();

        let inputs = Value::new(&x);
        let weights = Value::new(&w);
        let bias = Value::new(&b);

        let mut linear= ScaleAdd::new(
            Dot::new(inputs.clone(), weights.clone()),
            bias
        );

        let w_binding = weights.grad();
        let curr_w_grad = w_binding.values();
        assert_eq!(curr_w_grad, w.values());

        let input_binding = inputs.grad();
        let input_grad = input_binding.values();
        assert_eq!(x.values(), input_grad); 

        linear.forward();

        let expected_output: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let output_binding = linear.output.borrow().clone();
        let outputs = output_binding.val();
        assert_eq!(outputs.values(), &expected_output); 

        let y_pred = linear.value();
        let _loss = mse(&y, &y_pred);
        let error = y.subtract(y_pred.clone()).unwrap();
        let expected_error: Vec<f64> = vec![9.0, 11.0, 13.0, 15.0, 17.0];   
        assert_eq!(error.values(), &expected_error); 

        /*
        linear.backward(error);
        let expected_w_grad = vec![215.0, 280.0, 345.0];
        assert_eq!(weights.grad().values(), &expected_w_grad);
 
        for item in inputs.grad().values() {
            let expected: f64 = 0.0;
            assert_eq!(item, &expected); 
        } */

    }

    #[test]
    fn test_linear_graph() {

        let x: NDArray<f64> = NDArray::load("data/linear_testing_data/inputs").unwrap();
        let y: NDArray<f64> = NDArray::load("data/linear_testing_data/outputs").unwrap();
        let w: NDArray<f64> = NDArray::load("data/linear_testing_data/weights").unwrap();
        let b: NDArray<f64> = NDArray::load("data/linear_testing_data/bias").unwrap();

        // temp loss vars
        let mut first_loss = 0.0; 
        let mut curr_loss = 0.0;

        let inputs = Value::new(&x);
        let mut weights = Value::new(&w); // this needs to be mutable
        let mut biases = Value::new(&b);
        let learning_rate = 0.01; 

        let mut linear = ScaleAdd::new(
            Dot::new(inputs.clone(), weights.clone()),
            biases.clone()
        );


        for epoch in 0..10 {

            linear.forward();

            let y_pred = linear.value();
            let loss = mse(&y, &y_pred);
            let error = y_pred.subtract(y.clone()).unwrap();
            println!("Loss: {:?}", loss); 

            linear.backward(error);

            /* update weights */
            let w_grad = inputs.grad().scalar_mult(learning_rate/y_pred.size() as f64).unwrap();
            let dw = weights.val().subtract(w_grad).unwrap();
            weights.set_val(&dw); 

            /* update biases */
            let b_collapse = biases.grad().sum_axis(1).unwrap();
            let db = b_collapse.scalar_mult(learning_rate/y_pred.size() as f64).unwrap();
            biases.set_val(&db);

            if epoch == 0 {
                first_loss = loss.clone().unwrap(); 
            }

            if epoch == 9 {
                curr_loss = loss.unwrap();
            }

        }

        let loss_condition = curr_loss < first_loss; 
        assert_eq!(loss_condition, true); 

    }
}
