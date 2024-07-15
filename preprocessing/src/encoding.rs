use ndarray::ndarray::NDArray;


pub struct OneHotEncoding {
    input_column: NDArray<f64>,
    encoded_values: NDArray<f64>,
    max_value: f64, 
    num_samples: f64,
}


impl OneHotEncoding {

    pub fn new(input_column: NDArray<f64>) -> Result<OneHotEncoding, String>  {

        if input_column.shape().dim(1) != 1 {
            return Err("Input col must be of size (N, 1)".to_string())
        }

        if input_column.rank() > 2 {
            return Err("Input col must be less than rank 2".to_string())
        }

        let max_value = input_column.values().iter().max_by(
            |a, b| a.total_cmp(b)
        ).unwrap();
        let max_index = *max_value + 1.0;

        Ok(Self {
            input_column: input_column.clone(),
            encoded_values: NDArray::new(vec![
                input_column.shape().dim(0),
                max_index as usize
            ]).unwrap(),
            max_value: max_index.clone(), 
            num_samples: input_column.shape().dim(0) as f64
        })
    }

    pub fn max_value(&self) -> f64 {
        self.max_value
    }

    pub fn num_samples(&self) -> f64 {
        self.num_samples
    }

    pub fn transform(&mut self) -> &NDArray<f64> {

        let mut row = 0;
        let col_stride = self.encoded_values.shape().dim(1);
        for idx in self.input_column.values() {
            let index = (idx + row as f64) as usize; 
            let _ = self.encoded_values.set_idx(index, 1.0);
            row += col_stride;
        }

        &self.encoded_values
    }

}
