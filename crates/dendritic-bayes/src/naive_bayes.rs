use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;
use dendritic_metrics::activations::*;
use dendritic_metrics::loss::*;
use dendritic_metrics::utils::*; 
use std::fs;
use std::collections::HashSet;
use crate::shared::*;

#[derive(Debug)]
pub struct NaiveBayes {
    pub features: NDArray<f64>,
    pub outputs: NDArray<f64>,
    pub frequencies: Vec<NDArray<f64>>,
    pub likelihoods: Vec<NDArray<f64>>
}

impl NaiveBayes {

    /// Create new instance of naive bayes model
    pub fn new(
        features: &NDArray<f64>, 
        outputs: &NDArray<f64>) -> Result<NaiveBayes, String> {

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
            frequencies: Vec::new(), 
            likelihoods: Vec::new()
        })
    }


    /// Create frequency table of values
    pub fn frequency_table(
        &self,
        feature: NDArray<f64>,
        class_idxs: Vec<Vec<usize>>) -> Result<NDArray<f64>, String> {

        let feature_col = feature.shape().dim(1);
        let feature_row = feature.shape().dim(0);

        if feature_row != self.outputs.shape().dim(0) { 
            let msg = "Rows of feature must match rows of output";
            return Err(msg.to_string());
        }

        if feature_col != 1 {
            let msg = "Feature to frequency table must be shape (N, 1)";
            return Err(msg.to_string());
        }

        let mut idx = 0; 
        let feat_unique = feature.unique();
        let mut freq_table: NDArray<f64> = NDArray::new(
            vec![feat_unique.len(),class_idxs.len() + 1]
        ).unwrap();

        for val in &feat_unique {

            let feat_idxs = feature.value_indices(*val);
            freq_table.set_idx(idx, *val).unwrap();

            for cls in &class_idxs {
                let s1: HashSet<_> = cls.iter().cloned().collect();
                let s2: HashSet<_> = feat_idxs.iter().cloned().collect();
                let common: HashSet<_> = s1.intersection(&s2)
                    .cloned()
                    .collect();

                idx += 1;
                freq_table.set_idx(idx, common.len() as f64);
            }
            idx += 1;
        }

        Ok(freq_table)
    }

    /// Create likelihood of values based on row samples
    pub fn likelihood_table(
        &self, 
        freq_table: NDArray<f64>) -> NDArray<f64> {

        let mut class_idx = 0;
        let mut idx = 0;
        let mut table_vals: Vec<f64> = Vec::new(); 
        let cols = freq_table.shape().dim(1);
        let rows = freq_table.shape().dim(0); 
        let class_counts = class_idxs(&self.outputs);

        for row in 0..freq_table.shape().dim(0) {
            let item = freq_table.axis(0, row).unwrap();
        }


        for col in 0..cols {

            let item = freq_table.axis(1, col).unwrap();
            if col == 0 {
                let mut item_vals = item.values().clone();
                table_vals.append(&mut item_vals);
            } else {
                let class_cnt = class_counts[class_idx].len() as f64;
                let divide_class = item.scalar_div(class_cnt).unwrap();
                let mut vals = divide_class.values().clone();
                table_vals.append(&mut vals);
                class_idx += 1;

            } 
        }

        NDArray::array(
            vec![
                freq_table.shape().dim(1),
                freq_table.shape().dim(0)
            ],
            table_vals
        ).unwrap()
    }


    /// Feature prior probability utility
    pub fn feature_prior_probability(
        &self,
        feature_idx: usize,
        feature_value: f64) -> f64 {

        let rows = self.outputs.shape().dim(0);
        let selected_feature = self.features.axis(1, feature_idx).unwrap();
        let feature_vals = selected_feature.values();
        let count = feature_vals.iter().filter(
            |&&x| x == feature_value
        ).count();

        count as f64 / rows as f64

    }

    /// Predict likelihood of feature occurring
    pub fn predict_feature(
        &mut self, 
        feature_col: usize,
        value: f64, 
        class: f64) -> f64 {

        /* count number of features associated with class */
        let class_indices = class_idxs(&self.outputs);
        let freq_table = self.frequency_table(
            self.features.axis(1, feature_col).unwrap(),
            class_indices.clone()
        ).unwrap();
        let lh_table = self.likelihood_table(freq_table.clone());

        self.frequencies.push(freq_table);
        self.likelihoods.push(lh_table.clone());

        /* search first col of lh table */
        let mut row_idx = 0;
        let feature = lh_table.axis(0, 0).unwrap();
        for (idx, feat) in feature.values().iter().enumerate() {
            if *feat == value {
               row_idx = idx;  
            }
        }

        /* calculate bayes theorem with likelihood and frequency table */
        let target_class = class_indices[class as usize].len();
        let row_select = lh_table.axis(1, row_idx).unwrap();
        let feature_occurrence = row_select.values()[class as usize + 1];
        feature_occurrence 
    }


    /// Fit naive bayes model to all feature probabilities
    pub fn fit(&mut self, data: NDArray<f64>) -> usize {

        let mut largest_prob: f64 = 0.0;
        let mut predict_class: usize = 0;
        let class_count = class_idxs(&self.outputs);
        for class in 0..class_count.len() {

            let class_prob = class_probabilities(
                &self.outputs,
                class_idxs(&self.outputs)
            )[class]; 

            let mut sum = 1.0;
            for (idx, item) in data.values().iter().enumerate() {

                let predict = self.predict_feature(
                    idx, 
                    *item, 
                    class as f64
                );
                sum *= predict;
            }

            if sum * class_prob > largest_prob {
                largest_prob = sum * class_prob; 
                predict_class = class;
            }
 
        }

        predict_class
    }
    

}
