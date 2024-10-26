use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*; 

/// Apply vector function on ndarray value
pub fn apply(
    value: NDArray<f64>,
    axis: usize, 
    loss_function: fn(value: NDArray<f64>) -> NDArray<f64>) -> NDArray<f64> {

    let mut feature_vec: Vec<f64> = Vec::new();
    let shape = value.shape().dim(axis);
    for idx in 0..shape {
        let axis_value = value.axis(axis, idx).unwrap();
        let axis_loss = (loss_function)(axis_value);
        let mut axis_applied = axis_loss.values().clone(); 
        feature_vec.append(&mut axis_applied);
    }

    let result: NDArray<f64> = NDArray::array(
        value.shape().values(),
        feature_vec.clone()
    ).unwrap();

    result
}

/// Gini impurity for tree based models
pub fn gini_impurity(y: NDArray<f64>) -> f64 {
    let unique_samples = y.unique();
    let n = unique_samples.len();
    let mut gini = 0.0; 
    for idx in 0..n {
        let pi = unique_samples[idx];
        let pi_count = y.values().iter().filter(|&n| *n == pi).count();
        let val = pi_count as f64 / y.size() as f64;
        gini += val.powf(2.0);
    }
    1.0 - gini
}

/// Entropy calculation for tree based models
pub fn entropy(y: NDArray<f64>) -> f64 { 
    let mut ent = 0.0;
    let class_labels = y.unique();
    for class in &class_labels {

        let cls_count = y.values()
            .iter()
            .filter(|&x| x == class)
            .count();

        let p_cls = cls_count as f64 / y.size() as f64;
        ent += -p_cls * p_cls.log2();
    }
    ent
}

