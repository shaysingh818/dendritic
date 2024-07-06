use ndarray::ndarray::NDArray;
use ndarray::ops::*;


pub fn standard_scalar(input: NDArray<f64>) -> Result<NDArray<f64>, String>  {
   
    if input.rank() < 2 {
       return Err(
           "Standard Scalar: Must be with rank 2 or higher".to_string()
        ); 
    } 

    let mean_vals = input.mean(1).unwrap();
    let stdev_vals = input.stdev(1).unwrap();
    let shape = input.shape().dim(1);
    let mut feature_vec: Vec<f64> = Vec::new();

    for idx in 0..shape {
        let axis_vals = input.axis(1, idx).unwrap();
        let mean_val = mean_vals[idx];
        let stdev_val = stdev_vals[idx];
        let subtract_mean = axis_vals.scalar_subtract(mean_val).unwrap();
        let div_stdev = subtract_mean.scalar_div(stdev_val).unwrap();
        let mut dev_vals = div_stdev.values().clone();
        feature_vec.append(&mut dev_vals);
    }

    let temp: NDArray<f64> = NDArray::array(
        vec![shape, input.shape().dim(0)],
        feature_vec
    ).unwrap();

    let result = temp.transpose().unwrap();
    Ok(result)
}


pub fn min_max_scalar(input: NDArray<f64>) -> Result<NDArray<f64>, String> {

    if input.rank() < 2 {
       return Err(
           "MinMax Scalar: Must be with rank 2 or higher".to_string()
        ); 
    }

    let shape = input.shape().dim(1);
    let mut feature_vec: Vec<f64> = Vec::new();

    for idx in 0..shape {

        let axis_vals = input.axis(1, idx).unwrap();

        let min = axis_vals.values().iter().min_by(
            |a, b| a.total_cmp(b)
        ).unwrap();

        let max = axis_vals.values().iter().max_by(
            |a, b| a.total_cmp(b)
        ).unwrap();

        let subtract_min = axis_vals.scalar_subtract(*min).unwrap();
        let min_max = max - min;
        let div = subtract_min.scalar_div(min_max).unwrap();
        let mut div_values = div.values().clone();
        feature_vec.append(&mut div_values);
    }


    let temp: NDArray<f64> = NDArray::array(
        vec![shape, input.shape().dim(0)],
        feature_vec
    ).unwrap();

    let result = temp.transpose().unwrap();
    Ok(result)
}
