use steepgrad::ndarray;
use steepgrad::regression; 


#[cfg(test)]
mod logistic_tests {

    use crate::regression::logistic::Logistic;
    use crate::ndarray::ndarray::NDArray;
    use crate::ndarray::ops::*;

    #[test]
    fn test_logistic() {

        let x: NDArray<f64> = NDArray::load("data/logistic_testing_data/inputs").unwrap();
        let y: NDArray<f64> = NDArray::load("data/logistic_testing_data/outputs").unwrap();

        let mut model = Logistic::new(x, y, 0.01).unwrap();
        model.train(5000, false, 0);

        let loss = model.loss(); 
        let loss_condition = loss < 0.1; 
        assert_eq!(loss_condition, true);  

    }

    #[test]
    fn test_stochastic_logistic() {

        let x: NDArray<f64> = NDArray::load("data/logistic_testing_data/inputs").unwrap();
        let y: NDArray<f64> = NDArray::load("data/logistic_testing_data/outputs").unwrap();

        let mut model = Logistic::new(x, y, 0.01).unwrap();
        model.train(5000, false, 3);

        let loss = model.loss(); 
        let loss_condition = loss < 0.1; 
        assert_eq!(loss_condition, true);  

    }


    #[test]
    fn test_logistic_save_load() {

        let x: NDArray<f64> = NDArray::load("data/logistic_testing_data/inputs").unwrap();
        let y: NDArray<f64> = NDArray::load("data/logistic_testing_data/outputs").unwrap();

        let mut model = Logistic::new(x, y, 0.01).unwrap();
        model.train(5000, false, 3);

        let loss = model.loss(); 
        let loss_condition = loss < 0.1; 
        assert_eq!(loss_condition, true);  
 

    }


}