use rand::prelude::*; 
use ndarray::ndarray::NDArray;
use ndarray::ops::*;


pub struct Bootstrap {
    n_bootstraps: usize,
    num_features: usize,
    sample_size: usize,
    x_train: NDArray<f64>,
    datasets: Vec<NDArray<f64>>
}


impl Bootstrap {

    pub fn new(
        n_bootstraps: usize,
        num_features: usize,
        sample_size: usize,
        x_train: NDArray<f64>
    ) -> Bootstrap {

        Bootstrap {
            n_bootstraps: n_bootstraps,
            num_features: num_features,
            sample_size: sample_size,
            x_train: x_train,
            datasets: Vec::new()
        }

    }

    pub fn datasets(&self) -> &Vec<NDArray<f64>> {
        &self.datasets
    }

    pub fn n_bootstraps(&self) -> usize {
        self.n_bootstraps
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn sample_size(&self) -> usize {
        self.sample_size
    }

    pub fn generate(&mut self) {
        for item in 0..self.n_bootstraps {
            let dataset = self.sample(self.sample_size);
            self.datasets.push(dataset);
        }
    }


    pub fn feature_sub_select(&self)  -> NDArray<f64> {

        let num_cols = self.x_train.shape().dim(1)-1; 
        let mut features: Vec<usize> = Vec::new();
        for feature in 0..self.num_features {
            let rand_col = rand::thread_rng().gen_range(0..num_cols);
            features.push(rand_col);
        }

        features.push(num_cols);
        let dataset = self.x_train.select_axis(1, features).unwrap();
        dataset
    }


    pub fn sample(&self, sample_size: usize) -> NDArray<f64> {

        let rows = self.x_train.shape().dim(0);
        let mut values: Vec<f64> = Vec::new();
        for row in 0..sample_size {
            let rand_row = rand::thread_rng().gen_range(0..rows);
            let selected_row = self.x_train.axis(0, rand_row).unwrap();
            let mut row_values = selected_row.values().to_vec();
            values.append(&mut row_values);
        }

        let sample: NDArray<f64> = NDArray::array(
            vec![
                sample_size, 
                self.x_train.shape().dim(1)
            ],
            values
        ).unwrap();
        sample
    }

}
