use steepgrad::regression;
use steepgrad::ndarray;
use steepgrad::loss;

#[cfg(test)]
mod linear {

    use crate::ndarray::ndarray::NDArray;
    use crate::ndarray::ops::*;
    use crate::loss::mse::*;
    use crate::regression::linear::base::Linear;
    use crate::regression::linear::multiple::MultipleLinear;
    use crate::regression::linear::simple::SimpleLinear;

    #[test]
    fn test_single_linear_model() {

        let x: NDArray<f64> = NDArray::array(
            vec![5, 1], 
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        ).unwrap();

        let y: NDArray<f64> = NDArray::array(
            vec![5, 1], 
            vec![2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap();

        let mut model = SimpleLinear::new(x, y, 0.001).unwrap();
        model.train(500, false); 

    }

    #[test]
    fn multi_linear_model() {

        let x: NDArray<f64> = NDArray::array(
            vec![5, 3], 
            vec![
                1.0, 2.0, 3.0,
                2.0, 3.0, 4.0, 
                3.0, 4.0, 5.0,
                4.0, 5.0, 6.0, 
                5.0, 6.0, 7.0
            ]
        ).unwrap();

        let y: NDArray<f64> = NDArray::array(
            vec![5, 1], 
            vec![10.0, 12.0, 14.0, 16.0, 18.0]
        ).unwrap();

        let mut model = MultipleLinear::new(x.clone(), y.clone(), 0.01).unwrap();
        model.set_loss(mse); 

        let pre_train_results = model.predict(x.clone()).unwrap();
        let loss_before_train = mse(y.clone(), pre_train_results).unwrap();
        assert_eq!(loss_before_train, 204.0);

        model.train(1000, false);

        let result = model.predict(x).unwrap();
        let loss_after_train = mse(y.clone(), result).unwrap();
        let train_loss_results = loss_after_train < 0.01;
        assert_eq!(train_loss_results, true); 

    }
}