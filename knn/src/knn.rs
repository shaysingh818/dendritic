use std::collections::HashMap;
use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use crate::distance::*;
use crate::utils::*;


#[derive(Debug)]
pub struct KNN {
    pub features: NDArray<f64>,
    pub outputs: NDArray<f64>,
    pub k: usize,
    distance_metric: fn(
        y1: &NDArray<f64>, 
        y2: &NDArray<f64>
    ) -> Result<f64, String>
}


impl KNN {

    pub fn fit(
        features: &NDArray<f64>, 
        outputs: &NDArray<f64>,
        k: usize,
        distance_metric: fn(
            y1: &NDArray<f64>, 
            y2: &NDArray<f64>) -> Result<f64, String>
        ) -> Result<KNN, String> {

        let feature_rows = features.shape().dim(0);
        let output_rows  = outputs.shape().dim(0);
        if feature_rows != output_rows {
            return Err(
                "Feature rows must match output rows".to_string()
            );
        }

        Ok(Self {
            features: features.clone(),
            outputs: outputs.clone(),
            k: k, 
            distance_metric: distance_metric
        })
    }

    
    pub fn predict_sample(&self, point: &NDArray<f64>) -> f64 {

        let distances = calculate_distances(
            self.distance_metric,
            &self.features,
            point
        ).unwrap();

        let mut row_idxs: Vec<usize> = Vec::new();
        for item in 0..self.k {
            let dist = distances[item];
            let row = distances[item].1;
            row_idxs.push(row); 
        }

        let mut max_cnt = 0;
        let mut max_class = 0.0;
        let targets = self.outputs.indice_query(row_idxs).unwrap();
        let targets_unique = targets.unique();
        for target in &targets_unique {
            let cnt = targets.values().iter().filter(|&n| *n == *target).count();
            if cnt > max_cnt {
                max_cnt = cnt;
                max_class = *target;
            }
            
        }

        max_class
    }


    pub fn predict(&self, point: &NDArray<f64>) -> NDArray<f64> {

        let mut preds: Vec<f64> =Vec::new();
        let rows = point.shape().dim(0); 
        for row in 0..rows {
            let item = point.axis(0, row).unwrap();
            let pred = self.predict_sample(&item); 
            preds.push(pred);
        }

        NDArray::array(vec![preds.len(), 1], preds).unwrap()
    }

}


#[derive(Debug)]
pub struct KNNRegressor {
    pub features: NDArray<f64>,
    pub outputs: NDArray<f64>,
    pub k: usize,
    distance_metric: fn(
        y1: &NDArray<f64>, 
        y2: &NDArray<f64>
    ) -> Result<f64, String>
}


impl KNNRegressor {

    pub fn fit(
        features: &NDArray<f64>, 
        outputs: &NDArray<f64>,
        k: usize,
        distance_metric: fn(
            y1: &NDArray<f64>, 
            y2: &NDArray<f64>) -> Result<f64, String>
        ) -> Result<KNNRegressor, String> {

        let feature_rows = features.shape().dim(0);
        let output_rows  = outputs.shape().dim(0);
        if feature_rows != output_rows {
            return Err(
                "Feature rows must match output rows".to_string()
            );
        }

        Ok(Self {
            features: features.clone(),
            outputs: outputs.clone(),
            k: k, 
            distance_metric: distance_metric
        })
    }


    pub fn predict_sample(&self, point: &NDArray<f64>) -> f64 {

        let distances = calculate_distances(
            self.distance_metric,
            &self.features,
            point
        ).unwrap();

        let mut row_idxs: Vec<usize> = Vec::new();
        for item in 0..self.k {
            let dist = distances[item];
            let row = distances[item].1;
            row_idxs.push(row); 
        }

        let mut max_cnt = 0;
        let mut max_class = 0.0;
        let targets = self.outputs.indice_query(row_idxs).unwrap();
        targets.avg()
    }


    pub fn predict(&self, point: &NDArray<f64>) -> NDArray<f64> {

        let mut preds: Vec<f64> =Vec::new();
        let rows = point.shape().dim(0); 
        for row in 0..rows {
            let item = point.axis(0, row).unwrap();
            let pred = self.predict_sample(&item); 
            preds.push(pred);
        }

        NDArray::array(vec![preds.len(), 1], preds).unwrap()
    }

}
