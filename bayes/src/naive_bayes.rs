use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use metrics::activations::*;
use metrics::loss::*;
use metrics::utils::*; 
use std::fs;
use std::collections::HashSet; 

#[derive(Debug)]
pub struct NaiveBayes {
    pub features: NDArray<f64>,
    pub outputs: NDArray<f64>
}

impl NaiveBayes {

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
        })
    }


    pub fn class_idxs(&self) -> Vec<Vec<usize>> {

        let mut class_indices: Vec<Vec<usize>> = Vec::new();
        let y_target = self.outputs.unique();
        for item in &y_target {
            let indices = self.outputs.value_indices(*item);
            class_indices.push(indices);
        }

        class_indices
    }

    pub fn class_probabilities(&self) -> Vec<f64> {
        let mut probabilities: Vec<f64> = Vec::new();
        let class_indices = self.class_idxs();
        for item in &class_indices {
            let val = item.len() as f64 /self.outputs.size() as f64;
            probabilities.push(val as f64);
        }
        probabilities
    }

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

    pub fn likelihood_table(
        &self, 
        freq_table: NDArray<f64>) -> NDArray<f64> {

        let mut idx = 0;
        let mut table_vals: Vec<f64> = Vec::new(); 
        let cols = freq_table.shape().dim(1);
        let rows = freq_table.shape().dim(0); 
        let class_counts = self.class_idxs();
        let mut class_idx = 0;

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
                freq_table.shape().dim(0),
                freq_table.shape().dim(1)
            ],
            table_vals
        ).unwrap().transpose().unwrap()
    }


    pub fn predict_feature(
        &self, 
        feature_col: usize,
        value: f64, 
        class: f64) -> f64 {

        /* count number of features associated with class */
        let class_indices = self.class_idxs();
        let freq_table = self.frequency_table(
            self.features.axis(1, feature_col).unwrap(),
            class_indices.clone()
        ).unwrap();
        let lh_table = self.likelihood_table(freq_table);

        /* search first col of lh table */
        let mut row_idx = 0;
        let feature = lh_table.axis(1, 0).unwrap();
        for (idx, feat) in feature.values().iter().enumerate() {
            if *feat == value {
               row_idx = idx;  
            }
        }

        /* calculate bayes theorem with likelihood and frequency table */
        let target_class = class_indices[class as usize].len();
        let row_select = lh_table.axis(0, row_idx).unwrap();
        let feature_occurrence = row_select.values()[class as usize + 1];
        feature_occurrence 
    }


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





    

}
