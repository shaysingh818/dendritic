
#[cfg(test)]
mod value_test {

    use dendritic_autodiff::tensor::Tensor;
    use ndarray::prelude::*; 
    use ndarray::{arr2};

    #[test]
    fn test_value_instance() {

        let a = 5.0; 
        let mut val = Tensor::new(&a); 

        assert_eq!(val.value(), &5.0);  
        assert_eq!(val.grad(), &5.0);

        val.set_value(10.0);
        val.set_grad(20.0); 

        assert_eq!(val.value(), &10.0);  
        assert_eq!(val.grad(), &20.0);

        let a_ndarray = arr2(&[[1.0, 2.0, 3.0]]);
        let val_ndarray = Tensor::new(&a_ndarray);

        assert_eq!(
            val_ndarray.value(), 
            arr2(&[[1.0, 2.0, 3.0]])
        ); 

        assert_eq!(
            val_ndarray.grad(), 
            arr2(&[[1.0, 2.0, 3.0]])
        );  
    }


    #[test]
    fn test_value_generics() {

        let a: f64 = 5.0;
        let b: usize = 10; 
        let c: i64 = 1;
        let char_val: &str = "testing";
        let array_val = arr2(&[[1.0], [2.0], [3.0]]); 

        let mut val_a: Tensor<f64> = Tensor::new(&a); 
        let mut val_b: Tensor<usize> = Tensor::new(&b); 
        let mut val_c: Tensor<i64> = Tensor::new(&c); 
        let mut val_d: Tensor<&str> = Tensor::new(&char_val); 
        let mut val_e = Tensor::new(&array_val);

        assert_eq!(val_a.grad(), &5.0);  
        assert_eq!(val_b.grad(), &10);  
        assert_eq!(val_c.grad(), &1);  
        assert_eq!(val_d.grad(), &"testing");  
        assert_eq!(
            val_e.grad(), 
            arr2(&[[1.0], [2.0], [3.0]])
        );  

    }


}
