use std::collections::HashSet; 
use ndarray::{Array, Array2, Axis};
use ndarray_stats::QuantileExt;


pub trait Preprocessor {

    /// Encode data and return type associated with trait type
    fn encode(&mut self) -> Array2<f64>;

    /// Decode data from encoded values with associated trait type
    fn decode(&mut self, data: &Array2<f64>) -> Array2<f64>;

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

        let encoded_shape = (self.data.nrows(), self.num_classes);
        let mut encoded: Array2<f64> = Array2::zeros(encoded_shape);
        for (idx, row) in self.data().iter().enumerate() {
            encoded[[idx, *row as usize]] = 1.0;
        }
        encoded
    }

    /// Decode function for decoding One Hot Encoded Values
    fn decode(&mut self, data: &Array2<f64>) -> Array2<f64> {

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

        let mut encoded: Array2<f64> = Array2::zeros(self.data.dim());
     
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
            encoded.index_axis_mut(Axis(1), idx).assign(&feature_col); 
        }

        encoded
    }
    
    /// Decode function for decoding One Hot Encoded Values
    fn decode(&mut self, data: &Array2<f64>) -> Array2<f64> {
        let mean = Array::from_vec(self.mean());
        let std_dev = Array::from_vec(self.stdev());
        data * std_dev + mean
    }

}



#[derive(Debug, Clone)]
pub struct MinMaxScalar {
    
    /// Raw data passed to the encoder
    data: Array2<f64>,

    /// Vector of minimum values associated with each feature
    min_range: Vec<f64>,
    
    /// Vector of maximum values associated with each feature
    max_range: Vec<f64>

}


impl MinMaxScalar {

    pub fn new(data: &Array2<f64>) -> Result<Self, String> {
        Ok(MinMaxScalar { 
            data: data.clone(),
            min_range: vec![0.0; data.ncols()],
            max_range: vec![0.0; data.ncols()]
        })
    }

    pub fn data(&self) -> &Array2<f64> {
        &self.data
    }

    pub fn min_range(&self) -> &Vec<f64> {
        &self.min_range
    }

    pub fn max_range(&self) -> &Vec<f64> {
        &self.max_range
    }

}


impl Preprocessor for MinMaxScalar {

    /// Encode function for One Hot Encoding
    fn encode(&mut self) -> Array2<f64> {

        let mut encoded: Array2<f64> = Array2::zeros(self.data.dim());
        for (idx, col) in self.data.axis_iter(Axis(1)).enumerate() {
            
            let owned_col = col.to_owned();
            
            let min = owned_col.min().unwrap();
            let max = owned_col.max().unwrap();

            self.min_range[idx] = *min;
            self.max_range[idx] = *max;

            let min_vec = Array::from_elem(col.len(), *min);
            let max_vec = Array::from_elem(col.len(), *max);

            let subtract_min = owned_col - min_vec.clone();
            let min_max = max_vec - min_vec;
            let div = subtract_min / min_max;

            encoded.index_axis_mut(Axis(1), idx).assign(&div);
        }

        encoded
    }
    
    /// Decode function for decoding One Hot Encoded Values
    fn decode(&mut self, data: &Array2<f64>) -> Array2<f64> {

        if data.dim() != self.data.dim() {
            panic!("Encoded data does not much decoded data dimensions"); 
        }

        let mut decoded: Array2<f64> = Array2::zeros(self.data.dim());
        for (idx, col) in data.axis_iter(Axis(1)).enumerate() {

            let min_vec = Array::from_elem(
                col.len(), 
                self.min_range[idx]
            );

            let max_vec = Array::from_elem(
                col.len(), 
                self.max_range[idx]
            );

            let min_max = max_vec - min_vec.clone();
            let feature = min_max * col + min_vec;
            decoded.index_axis_mut(Axis(1), idx).assign(&feature);              
        }

        decoded
    }

}



#[cfg(test)]
mod preprocessing_tests {

    use ndarray::{arr2}; 
    use crate::preprocessing::processor::*;

    #[test]
    fn test_one_hot_encoding() {

        let x = arr2(&[
            [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0], [2.0], [2.0]
        ]);

        let encoded = arr2(&[
            [1.0,0.0,0.0],
            [1.0,0.0,0.0],
            [1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,1.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,1.0],
            [0.0,0.0,1.0],
            [0.0,0.0,1.0]
        ]);

        let mut one_hot = OneHotEncoding::new(&x).unwrap();
        let bad_on_hot = OneHotEncoding::new(
            &arr2(&[
                 [0.0, 0.0],
                 [0.0, 0.0]
            ])
        );

        assert_eq!(one_hot.num_classes(), 3); 
        assert_eq!(one_hot.num_samples(), 9); 
        assert_eq!(one_hot.data().dim(), x.dim());

        assert_eq!(
            bad_on_hot.unwrap_err().to_string(),
            "Input must be of size (N, 1)"
        );

        assert_eq!(one_hot.encode(), encoded);

        let decoded = one_hot.decode(&encoded);
        assert_eq!(decoded, x); 
 
    }


    #[test]
    fn test_standard_scalar() {

        let x = arr2(&[
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ]);

        let mut scalar = StandardScalar::new(&x).unwrap();

        assert_eq!(scalar.data(), x); 
        assert_eq!(scalar.data().dim(), x.dim());
        assert_eq!(scalar.mean().len(), 2); 
        assert_eq!(scalar.stdev().len(), 2);

        let encoded_data = scalar.encode(); 

        assert_eq!(
            encoded_data.mapv(|x| (x * 10000.0).round() / 10000.0),
            arr2(&[
                [-1.4142, -1.4142],
                [-0.7071, -0.7071],
                [ 0.0000,  0.0000],
                [ 0.7071,  0.7071],
                [ 1.4142,  1.4142]
            ])
        );

        assert_eq!(scalar.decode(&encoded_data), x);

    }


    #[test]
    fn test_min_max_scalar() {

        let x = arr2(&[
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ]);

        let expected = arr2(&[
            [0.0, 0.0],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.75, 0.75],
            [1.0, 1.0]
        ]); 

        let mut min_max = MinMaxScalar::new(&x).unwrap();

        assert_eq!(min_max.data().dim(), x.dim()); 
        assert_eq!(min_max.data(), &x);
        assert_eq!(min_max.min_range().len(), x.ncols());
        assert_eq!(min_max.max_range().len(), x.ncols());

        let encoded = min_max.encode();

        assert_eq!(min_max.min_range(), &vec![1.0, 2.0]); 
        assert_eq!(min_max.max_range(), &vec![5.0, 10.0]);
        assert_eq!(&encoded, expected);

        let decoded = min_max.decode(&encoded);
        assert_eq!(decoded, x);

    }


}
