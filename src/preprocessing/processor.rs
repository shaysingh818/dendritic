use std::collections::HashSet; 
use ndarray::{Array, Array2, ArrayView2, Axis};
use ndarray_stats::QuantileExt;


/// Trait for housing shared behavior of feature encoders
pub trait FeatureEncoder {
 
    /// Transform data for specific feature encoder
    fn transform(&mut self, data: &ArrayView2<f64>) -> Array2<f64>; 

    /// Decode transformed data from feature encoder
    fn inverse_transform(&self, data: &ArrayView2<f64>) -> Array2<f64>; 

}


/// One hot encoder
#[derive(Debug, Clone)]
pub struct OneHot {

    /// Num classes detected
    num_classes: usize, 

    /// Number of samples that were encoded
    num_samples: usize

}


impl OneHot {

    /// Create instance of OneHot Encoder.
    ///
    /// # Arguments
    ///
    /// * `data` - 2D NDArray with features.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dendritic::preprocessing::processor::*;
    ///
    /// let data = arr2(&[
    ///     [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0], [2.0], [2.0]
    /// ]);
    /// let mut one_hot = OneHot::new();
    /// let encoded  = one_hot.transform(&data.view());
    /// println!("Encoded: {:?}", encoded);
    /// ```
    pub fn new() -> Self {
        Self {
            num_classes: 0,
            num_samples: 0
        }

    }

    /// Retrieve number of classes associated with encoded data
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Retrieve number of samples associated with encoded data
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

}


impl FeatureEncoder for OneHot {

    fn transform(&mut self, data: &ArrayView2<f64>) -> Array2<f64> {

        if data.dim().1 != 1 {
            let msg = "Input must be of size (N, 1)";
            panic!("{}", msg.to_string());
        }
 
        let mut vals = HashSet::new();
        data.mapv(|x| vals.insert(x as usize)); 
        let num_classes = vals.len();
        self.num_classes = num_classes;

        let encoded_shape = (data.nrows(), num_classes);
        let mut encoded: Array2<f64> = Array2::zeros(encoded_shape);
        for (idx, row) in data.iter().enumerate() {
            encoded[[idx, *row as usize]] = 1.0;
            self.num_samples += 1; 
        }

        encoded
    }


    fn inverse_transform(&self, data: &ArrayView2<f64>) -> Array2<f64> {

        let mut decoded: Array2<f64> = Array2::zeros((data.nrows(), 1));
        for (idx, row) in data.axis_iter(Axis(0)).enumerate() {
            if let Some(col) = row.iter().position(|&x| x == 1.0) {
                decoded[[idx, 0]] = col as f64;
            }
        }

        decoded
    }

}


/// One hot encoder 
pub struct MinMax {

    /// Vector of minimum values associated with each feature
    min_range: Vec<f64>,
    
    /// Vector of maximum values associated with each feature
    max_range: Vec<f64>
}


impl MinMax {

    /// Create instance of MinMaxScaler.
    ///
    /// # Arguments
    ///
    /// * `data` - 2D NDArray with feature output column .
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dendritic::preprocessing::processor::*;
    ///
    /// let x = arr2(&[
    ///     [1.0, 2.0],
    ///     [2.0, 4.0],
    ///     [3.0, 6.0],
    ///     [4.0, 8.0],
    ///     [5.0, 10.0],
    /// ]);
    /// let mut scalar = MinMax::new();
    /// let encoded = scalar.transform(&x.view()); 
    /// println!("Encoded data: {:?}", encoded);
    /// ```
    pub fn new() -> Self {
        Self {
            min_range: vec![],
            max_range: vec![]
        }

    }

    /// Retrieve the minimum values for each feature
    pub fn min_range(&self) -> &Vec<f64> {
        &self.min_range
    }

    /// Retrieve the maximum values for each feature
    pub fn max_range(&self) -> &Vec<f64> {
        &self.max_range
    }

}


impl FeatureEncoder for MinMax {

    fn transform(&mut self, data: &ArrayView2<f64>) -> Array2<f64> {

        let mut encoded: Array2<f64> = Array2::zeros(data.dim());
        for (idx, col) in data.axis_iter(Axis(1)).enumerate() {
            
            let owned_col = col.to_owned();
            
            let min = owned_col.min().unwrap();
            let max = owned_col.max().unwrap();

            self.min_range.push(*min);
            self.max_range.push(*max);

            let min_vec = Array::from_elem(col.len(), *min);
            let max_vec = Array::from_elem(col.len(), *max);

            let subtract_min = owned_col - min_vec.clone();
            let min_max = max_vec - min_vec;
            let div = subtract_min / min_max;

            encoded.index_axis_mut(Axis(1), idx).assign(&div);
        }

        encoded
    }


    fn inverse_transform(&self, data: &ArrayView2<f64>) -> Array2<f64> {

        
        if data.dim() != data.dim() {
            panic!("Encoded data does not much decoded data dimensions"); 
        }

        let mut decoded: Array2<f64> = Array2::zeros(data.dim());
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


/// Standard scalar for standarization of feature columns
#[derive(Debug, Clone)]
pub struct StandardScalar {
    
    /// Max value associated with encoder
    mean: Vec<f64>,

    /// Number of samples that were encoded
    standard_deviation: Vec<f64>

}


impl StandardScalar {

    /// Create instance of StandardScaler.
    ///
    /// # Arguments
    ///
    /// * `data` - 2D NDArray with feature output column .
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dendritic::preprocessing::processor::*;
    ///
    /// let x = arr2(&[
    ///     [1.0, 2.0],
    ///     [2.0, 4.0],
    ///     [3.0, 6.0],
    ///     [4.0, 8.0],
    ///     [5.0, 10.0],
    /// ]);
    /// let mut scalar = StandardScalar::new();
    /// let encoded = scalar.transform(&x.view()); 
    /// println!("Encoded data: {:?}", encoded);
    /// ```
    pub fn new() -> Self {
         
        Self {
            mean: vec![],
            standard_deviation: vec![]
        } 

    }

    /// Retrieve the mean values for each feature
    pub fn mean(&self) -> Vec<f64> { 
        self.mean.clone()
    }

    /// Retrieve the standard deviation values for each feature
    pub fn stdev(&self) -> Vec<f64> {
        self.standard_deviation.clone()
    }

}


impl FeatureEncoder for StandardScalar {

    /// Encode function for One Hot Encoding
    fn transform(&mut self, data: &ArrayView2<f64>) -> Array2<f64> {

        let mut encoded: Array2<f64> = Array2::zeros(data.dim());
     
        for (idx, col) in data.axis_iter(Axis(1)).enumerate() {

            self.mean.push(col.mean().unwrap());
            let mean_vec = Array::from_elem(
                data.nrows(), 
                col.mean().unwrap()
            );

            let mut diffs = col.to_owned() - mean_vec.clone();
            diffs.mapv_inplace(|x| x * x); 

            let variance = diffs.sum() / data.nrows() as f64;
            self.standard_deviation.push(variance.sqrt());
            let std_dev_vec = Array::from_elem(
                data.nrows(), 
                variance.sqrt()
            );

            let feature_col = (col.to_owned() - mean_vec.clone()) / std_dev_vec.clone();
            encoded.index_axis_mut(Axis(1), idx).assign(&feature_col); 
        }

        encoded
    }
    
    /// Decode function for decoding One Hot Encoded Values
    fn inverse_transform(&self, data: &ArrayView2<f64>) -> Array2<f64> {
        let mean = Array::from_vec(self.mean());
        let std_dev = Array::from_vec(self.stdev());
        data * std_dev + mean
    }

}


#[cfg(test)]
mod preprocessing_tests {

    use ndarray::{s, arr2}; 
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

        let mut one_hot = OneHot::new();
        let transformed = one_hot.transform(&x.view());

        assert_eq!(one_hot.num_classes(), 3); 
        assert_eq!(one_hot.num_samples(), x.nrows());

        assert_eq!(transformed.dim(), encoded.dim());
        assert_eq!(transformed, encoded);

        let decoded = one_hot.inverse_transform(&transformed.view()); 
        
        assert_eq!(decoded.dim(), x.dim());
        assert_eq!(decoded, x);


        let x1 = arr2(&[
            [1.0,0.0,0.0],
            [1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,1.0],
            [0.0,0.0,1.0],
            [0.0,0.0,1.0]
        ]);

        let x1_decoded = one_hot.inverse_transform(&x1.view());

        assert_eq!(
            x1_decoded, 
            arr2(&[
                 [0.0], [0.0], [1.0], 
                 [2.0], [2.0], [2.0]
            ])
        );
 
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

        let mut scalar = StandardScalar::new();

        let encoded = scalar.transform(&x.view());

        assert_eq!(scalar.mean().len(), 2); 
        assert_eq!(scalar.stdev().len(), 2);

        assert_eq!(
            encoded.mapv(|x| (x * 10000.0).round() / 10000.0),
            arr2(&[
                [-1.4142, -1.4142],
                [-0.7071, -0.7071],
                [ 0.0000,  0.0000],
                [ 0.7071,  0.7071],
                [ 1.4142,  1.4142]
            ])
        );

        assert_eq!(scalar.inverse_transform(&encoded.view()), x);

        let uneven = encoded.slice(s![0..3, ..]);
        let uneven_encoded = scalar.inverse_transform(&uneven);

        assert_eq!(uneven_encoded, x.slice(s![0..3, ..])); 
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

        let mut min_max = MinMax::new();
        let encoded = min_max.transform(&x.view());

        assert_eq!(min_max.min_range().len(), x.ncols());
        assert_eq!(min_max.max_range().len(), x.ncols());

        assert_eq!(min_max.min_range(), &vec![1.0, 2.0]); 
        assert_eq!(min_max.max_range(), &vec![5.0, 10.0]);

        assert_eq!(&encoded, expected);

        let decoded = min_max.inverse_transform(&encoded.view());
        assert_eq!(decoded, x);

    }


}
