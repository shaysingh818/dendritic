use std::fs;
use std::collections::HashSet; 
use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use metrics::activations::*;
use metrics::loss::*;
use metrics::utils::*; 
use crate::shared::*;


#[derive(Debug)]
pub struct GaussianNB {
    pub features: NDArray<f64>,
    pub outputs: NDArray<f64>,
    pub likelihoods: NDArray<f64>
}


impl GaussianNB {

    /// Get all likelihoods of gaussian naive bayes
    pub fn likelihoods(&self) -> NDArray<f64> {
        self.likelihoods.clone()
    }


    /// Save guassian bayes model parameters
    pub fn save(&self, filepath: &str) -> std::io::Result<()> {
        let likelihoods_file = format!("{}/likelihoods", filepath);
        fs::create_dir_all(filepath)?;
        self.likelihoods.save(&likelihoods_file).unwrap();
        Ok(())
    }


    /// Load guassian bayes model parameters
    pub fn load(
        filepath: &str, 
        features: &NDArray<f64>, 
        outputs: &NDArray<f64>) -> std::io::Result<GaussianNB> {

        let likelihoods_file = format!("{}/likelihoods", filepath);
        let load_likelihoods = NDArray::load(&likelihoods_file).unwrap();

        Ok(Self {
            features: features.clone(),
            outputs: outputs.clone(),
            likelihoods: load_likelihoods
        })
    }

    /// Create new instance of gaussian bayes model
    pub fn new(
        features: &NDArray<f64>, 
        outputs: &NDArray<f64>) -> Result<GaussianNB, String> {

        let feature_rows = features.shape().dim(0);
        let output_rows  = outputs.shape().dim(0);
        if feature_rows != output_rows {
            return Err(
                "Feature rows must match output rows".to_string()
            );
        }

        let mut instance = Self {
            features: features.clone(),
            outputs: outputs.clone(),
            likelihoods: NDArray::new(vec![0, 0]).unwrap()
        }; 
        instance.build_likelihoods();
        Ok(instance)
    }

    /// Build all likelihoods for gaussian naive bayes model
    pub fn build_likelihoods(&mut self) {

        let mut table_vals: Vec<f64> = Vec::new();
        let feature_cols = self.features.shape().dim(1);
        for col in 0..feature_cols {
            let item = self.features.axis(1, col).unwrap();
            let class_indices = class_idxs(&self.outputs);
            for class in &class_indices {
                let vals = item.indice_query(class.clone()).unwrap();
                let mean = vals.avg();
                let std_dev = vals.stdev_sample(1).unwrap();
                table_vals.push(mean);
                table_vals.push(std_dev[0]);
            }
        }

        self.likelihoods = NDArray::array(
            vec![class_idxs(&self.outputs).len(), feature_cols, 2],
            table_vals
        ).unwrap()
    }


    /// Predict likelihood for a given feature
    pub fn predict_feature(
        &mut self,
        feature_col: usize,
        value: f64, 
        class: f64) -> f64 {

        let likelihoods = self.likelihoods();
        let select_class = likelihoods.axis(1, class as usize).unwrap();
        let start_ft_idx = feature_col * 2;

        gaussian_probability(
            value,
            select_class.values()[start_ft_idx],
            select_class.values()[start_ft_idx + 1]
        )
    }

    /// Fit prediction for a given row sample
    pub fn fit_row(&mut self, x: NDArray<f64>) -> Result<f64, String> {
     
        if x.shape().dim(0) != self.features.shape().dim(1) {
            let msg = "row sample not equal to features column count";
            return Err(msg.to_string());
        }
        
        let mut largest_prob: f64 = 0.0;
        let mut predict_class: f64 = 0.0;
        let class_count = class_idxs(&self.outputs);
        for class in 0..class_count.len() {

            let mut sum = 1.0;
            let class_prob_calc = class_probabilities(
                &self.outputs,
                class_idxs(&self.outputs)
            );
            let class_prob = class_prob_calc[class as usize];

            for (idx, item) in x.values().iter().enumerate() {
                let pred = self.predict_feature(idx, *item, class as f64);
                sum *= pred; 
            }

            if sum * class_prob > largest_prob {
                largest_prob = sum * class_prob; 
                predict_class = class as f64;
            }
        }

        Ok(predict_class)
    }

    /// Fit all rows and give prediction for each feature
    pub fn fit(&mut self, x: NDArray<f64>) -> Result<NDArray<f64>, String> {
        if x.shape().dim(1) != self.features.shape().dim(1) {
            let msg = "row sample not equal to features column count";
            return Err(msg.to_string());
        }

        let mut preds: Vec<f64> = Vec::new();
        let rows = x.shape().dim(0); 
        for row in 0..rows {
            let item = x.axis(0, row).unwrap();
            let pred = self.fit_row(item).unwrap();
            preds.push(pred);
        }

        Ok(NDArray::array(
            vec![preds.len(), 1],
            preds
        ).unwrap())
    }
}

