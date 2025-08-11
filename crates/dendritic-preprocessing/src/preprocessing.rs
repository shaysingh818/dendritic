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
    max_value: f64,

    /// Number of samples that were encoded
    num_samples: f64

}


impl OneHotEncoding {
    
    pub fn new(data: Array2<f64>) -> Result<Self, String> {

        if data.dim().1 != 1 {
            let msg = "Input must be of size (N, 1)";
            return Err(msg.to_string());
        }

        let max_index = 0.00; 
        if let Some((max_i, max_val)) = data
            .indexed_iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
        {
            max_index = max_i as f64 + 1.0;
        } else {
            debug!("[OneHotEncoding]: No max value found"); 
        }

        Ok(Self {
            data: data,
            encoded_data: Array2::zeros(data.dim()),
            max_value: max_index, 
            num_samples: data.nrows()
        })

    }

}
