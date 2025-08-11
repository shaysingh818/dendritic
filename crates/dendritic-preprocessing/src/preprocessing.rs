use log::debug;
use std::collections::HashSet; 
use ndarray::{arr2, Array2, Axis};


pub trait Preprocessor {

    /// Encode data and return type associated with trait type
    fn encode(&mut self) -> Array2<f64>;

    /// Decode data from encoded values with associated trait type
    fn decode(&mut self) -> Array2<f64>;

}


pub struct OneHotEncoding {
    
    /// Raw data passed to the encoder
    data: Array2<f64>,

    /// Encoded data from the input
    encoded: Array2<f64>,

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
            encoded: Array2::zeros((data.dim().0, num_classes)),
            num_classes: num_classes, 
            num_samples: data.nrows()
        }) 

    }

    pub fn data(&self) -> Array2<f64> {
        self.data.clone()
    }

    pub fn encoded(&self) -> Array2<f64> {
        self.encoded.clone()
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

}


impl Preprocessor for OneHotEncoding {

    /// Encode function for One Hot Encoding
    fn encode(&mut self) -> Array2<f64> {
        println!("Still need to do");
        self.data.clone()
    }

    /// Decode function for decoding One Hot Encoded Values
    fn decode(&mut self) -> Array2<f64> {
        println!("Not done yet"); 
        self.data.clone()
    }

}
