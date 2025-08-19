use std::any::type_name; 
use ndarray::NdFloat; 
use ndarray::{arr2, Array2, Array, Axis};
use ndarray_stats::QuantileExt;

fn print_type_of<T>(_: &T) {
    println!("{}", type_name::<T>());
}


fn main() {

    let mut x = arr2(&[
        [1.0,2.0,30.0],
        [2.0,3.0,20.0],
        [3.0,10.0,10.0],
        [4.0,11.0,11.0],
        [5.0,12.0,8.0],
    ]);

    let mut x1: Array2<f64> = Array2::zeros(x.dim());

    for (idx, col) in x.axis_iter(Axis(1)).enumerate() {

        let owned_col = col.to_owned();

        let min = owned_col.min().unwrap();
        let max = owned_col.max().unwrap();

        let min_vec = Array::from_elem(col.len(), *min);
        let max_vec = Array::from_elem(col.len(), *max);

        let subtract_min = owned_col - min_vec.clone();
        let min_max = max_vec - min_vec;
        let div = subtract_min / min_max;

        x1.index_axis_mut(Axis(1), idx).assign(&div);
    }   





}
