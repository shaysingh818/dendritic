use log::debug;
use std::collections::HashSet; 
use ndarray::{arr2, Array, Array2, Axis};


pub trait Preprocessor {

    /// Encode data and return type associated with trait type
    fn encode(&mut self) -> Array2<f64>;

    /// Decode data from encoded values with associated trait type
    fn decode(&mut self, data: Array2<f64>) -> Array2<f64>;

}

#[derive(Debug, Clone)]
pub struct OneHotEncoding {
    
    /// Raw data passed to the encoder
    data: Array2<f64>,

    /// Max value associated with encoder
    num_classes: usize,

    /// Number of samples that were encoded
    num_samples: usize

}


impl OneHotEncoding {
    
    pub fn new(data: &Array2<f64>) -> Result<Self, String> {

        if data.dim().1 != 1 {
            let msg = "Input must be of size (N, 1)";
            return Err(msg.to_string());
        }

        
        let mut vals = HashSet::new();
        data.mapv(|x| vals.insert(x as usize)); 
        let num_classes = vals.len();

        Ok(Self {
            data: data.clone(),
            num_classes: num_classes, 
            num_samples: data.nrows()
        }) 

    }

    /// Retrieve data passed to one hot encoder
    pub fn data(&self) -> Array2<f64> {
        self.data.clone()
    }

    /// Retrieve the number of classes associated with data
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Retrieve the total number of samples being encoded
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

}


impl Preprocessor for OneHotEncoding {

    /// Encode function for One Hot Encoding
    fn encode(&mut self) -> Array2<f64> {

        let mut row: usize = 0;
        let encoded_shape = (self.data.nrows(), self.num_classes);
        let mut encoded: Array2<f64> = Array2::zeros(encoded_shape);
        for (idx, row) in self.data().iter().enumerate() {
            encoded[[idx, *row as usize]] = 1.0;
        }
        encoded
    }

    /// Decode function for decoding One Hot Encoded Values
    fn decode(&mut self, data: Array2<f64>) -> Array2<f64> {

        let mut decoded: Array2<f64> = Array2::zeros((data.nrows(), 1));
        for (idx, row) in data.axis_iter(Axis(0)).enumerate() {
            if let Some(col) = row.iter().position(|&x| x == 1.0) {
                decoded[[idx, 0]] = col as f64;
            }
        }
        decoded
    }

}


#[derive(Debug, Clone)]
pub struct StandardScalar {
    
    /// Raw data passed to the encoder
    data: Array2<f64>,

    /// Max value associated with encoder
    mean: Vec<f64>,

    /// Number of samples that were encoded
    standard_deviation: Vec<f64>

}


impl StandardScalar {

    pub fn new(data: &Array2<f64>) -> Result<Self, String> {
         
        Ok(Self {
            data: data.clone(),
            mean: vec![0.0; data.ncols()],
            standard_deviation: vec![0.0; data.ncols()]
        }) 

    }

    pub fn data(&self) -> Array2<f64> {
        self.data.clone()
    }

    pub fn mean(&self) -> Vec<f64> { 
        self.mean.clone()
    }

    pub fn stdev(&self) -> Vec<f64> {
        self.standard_deviation.clone()
    }

}


impl Preprocessor for StandardScalar {

    /// Encode function for One Hot Encoding
    fn encode(&mut self) -> Array2<f64> {

        let encoded: Array2<f64> = Array2::zeros(self.data.dim());
     
        for (idx, col) in self.data.axis_iter(Axis(1)).enumerate() {

            self.mean[idx] = col.mean().unwrap();
            let mean_vec = Array::from_elem(
                self.data.nrows(), 
                self.mean[idx]
            );

            let mut diffs = col.to_owned() - mean_vec.clone();
            diffs.mapv_inplace(|x| x * x); 

            let variance = diffs.sum() / self.data.nrows() as f64;
            self.standard_deviation[idx] = variance.sqrt();
            let std_dev_vec = Array::from_elem(
                self.data.nrows(), 
                self.standard_deviation[idx]
            );

            let feature_col = (col.to_owned() - mean_vec.clone()) / std_dev_vec.clone();
            println!("{:?}", feature_col); 
        }




        Array2::zeros(self.data.dim())
    }

    /// Decode function for decoding One Hot Encoded Values
    fn decode(&mut self, data: Array2<f64>) -> Array2<f64> {
        Array2::zeros(self.data.dim())
    }

}
