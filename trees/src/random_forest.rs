use std::fs; 
use std::fs::{File}; 
use std::io::{BufWriter, Write};
use std::collections::BTreeMap;

use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use crate::node::*;
use crate::utils::*;
use crate::bootstrap::*;
use crate::decision_tree::*; 
use crate::decision_tree_regressor::*; 

pub struct RandomForestClassifier {
    max_depth: usize, 
    samples_split: usize,
    n_trees: usize,
    num_features: usize,
    metric_function: fn(x: NDArray<f64>) -> f64,
    trees: Vec<DecisionTreeClassifier>
}


impl RandomForestClassifier {

    pub fn new(
        max_depth: usize,
        samples_split: usize,
        n_trees: usize,
        num_features: usize, 
        metric_function: fn(x: NDArray<f64>) -> f64
    ) -> RandomForestClassifier {

        RandomForestClassifier {
            max_depth: max_depth,
            samples_split: samples_split,
            n_trees: n_trees,
            num_features: num_features,
            metric_function: metric_function,
            trees: Vec::new()
        }

    }


    pub fn max_depth(&self) -> usize {
        self.max_depth
    }


    pub fn samples_split(&self) -> usize {
        self.samples_split
    }

    
    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    
    pub fn num_features(&self) -> usize {
        self.num_features
    }
    
    pub fn trees(&self) -> &Vec<DecisionTreeClassifier> {
        &self.trees
    }


    pub fn bootstrap_trees(
        &mut self,
        features: &NDArray<f64>,
        target: &NDArray<f64>) -> Result<(), String> {

        if self.num_features > features.shape().dim(1) {
            let msg = "Random Forest: Number of bootstrap features too large";
            return Err(msg.to_string());
        }

        let mut bs = Bootstrap::new(
            self.n_trees,
            self.num_features,
            features.shape().dim(0),
            features.clone()
        );
        bs.generate();

        let mut counter = 0; 
        for item in bs.datasets() {

           let mut tree = DecisionTreeClassifier::new(
               self.max_depth,
               self.samples_split,
               self.metric_function
           );
           tree.fit(&item, &target);
           self.trees.push(tree);
           counter += 1; 
        }

        Ok(())
    }


    pub fn load_trees(
        &mut self,
        filepath: &str,
        features: &NDArray<f64>,
        target: &NDArray<f64>) {

        let paths = fs::read_dir(filepath).unwrap();
        let mut tree_count = 0;

        for path in paths {
            let path_str = path.unwrap().path().display().to_string();
            let mut dt =  DecisionTreeClassifier::load(
                &path_str,
                self.max_depth, 
                self.samples_split,
                self.metric_function
            ); 
            dt.fit(features, target);
            self.trees.push(dt);
            tree_count += 1; 
        }
        self.n_trees = tree_count;
    }


    pub fn fit_loaded(
        &mut self,
        features: &NDArray<f64>,
        target: &NDArray<f64>,
        filepath: &str) {

        self.load_trees(
            filepath,
            features,
            target
        );

    }

    pub fn fit(
        &mut self, 
        features: &NDArray<f64>, 
        target: &NDArray<f64>) {

        self.bootstrap_trees(features, target).unwrap();
    }


    pub fn predict(&self, features: NDArray<f64>) -> NDArray<f64> {
        let rows = features.shape().dim(0);
        let mut y_predictions: Vec<f64> = Vec::new(); 
        for item in 0..rows {
            let row = features.axis(0, item).unwrap(); 
            let mut predictions = Vec::new(); 
            for tree in self.trees() {
                let y_pred = tree.prediction(
                    row.clone(), tree.root().clone()
                );
                predictions.push(y_pred); 
            }

            let prediction = self.frequency_check(predictions).unwrap();
            y_predictions.push(prediction as f64); 
        }

        NDArray::array(vec![rows, 1], y_predictions).unwrap() 
    }


    pub fn frequency_check(&self, values: Vec<f64>) -> Option<usize> {
        let mut frequency_map = BTreeMap::new(); 
        for &item in &values {
            let temp = item as usize; 
            *frequency_map.entry(temp).or_insert(0) += 1;
        }
        frequency_map.into_iter().max_by_key(
            |&(_, count)| count
        ).map(|(item, _)|item)
    }


    pub fn save(&self, filepath: &str) -> std::io::Result<()> {

        let mut counter = 0;
        for tree in &self.trees {
 
            let tree_path = format!(
                "{}/tree_{}", 
                filepath, counter
            );
            tree.save(&tree_path).unwrap();
            counter += 1; 
        }
        Ok(())
    }


    pub fn load(
        max_depth: usize,
        samples_split: usize,
        metric_function: fn(x: NDArray<f64>) -> f64) -> RandomForestClassifier {
      
        RandomForestClassifier {
            max_depth: max_depth,
            samples_split: samples_split,
            n_trees: 0,
            num_features: 0,
            metric_function: metric_function,
            trees: Vec::new()
        }
    }

}


pub struct RandomForestRegressor {
    max_depth: usize, 
    samples_split: usize,
    n_trees: usize,
    num_features: usize,
    loss_function: fn(
        y_true: &NDArray<f64>, 
        y_pred: &NDArray<f64>) -> Result<f64, String>,
    trees: Vec<DecisionTreeRegressor>
}


impl RandomForestRegressor {


    pub fn new(
        max_depth: usize,
        samples_split: usize,
        n_trees: usize,
        num_features: usize,
        loss_function: fn(
            y_true: &NDArray<f64>, 
            y_pred: &NDArray<f64>) -> Result<f64, String>,
    ) -> RandomForestRegressor {

        RandomForestRegressor {
            max_depth: max_depth,
            samples_split: samples_split,
            n_trees: n_trees,
            num_features: num_features,
            loss_function: loss_function,
            trees: Vec::new()
        }

    }


    pub fn max_depth(&self) -> usize {
        self.max_depth
    }


    pub fn samples_split(&self) -> usize {
        self.samples_split
    }

    
    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    
    pub fn trees(&self) -> &Vec<DecisionTreeRegressor> {
        &self.trees
    }


    pub fn save(&self, filepath: &str) -> std::io::Result<()> {

        let mut counter = 0;
        for tree in &self.trees {
 
            let tree_path = format!(
                "{}/tree_{}", 
                filepath, counter
            );
            tree.save(&tree_path).unwrap();
            counter += 1; 
        }
        Ok(())
    }

    pub fn bootstrap_trees(
        &mut self,
        features: &NDArray<f64>,
        target: &NDArray<f64>) -> Result<(), String> {

        if self.num_features > features.shape().dim(1) {
            let msg = "Random Forest: Number of bootstrap features too large";
            return Err(msg.to_string());
        }

        let mut bs = Bootstrap::new(
            self.n_trees,
            self.num_features,
            features.shape().dim(0),
            features.clone()
        );
        bs.generate(); 

        let mut counter = 0; 
        for item in bs.datasets() {

           let mut tree = DecisionTreeRegressor::new(
               self.max_depth,
               self.samples_split,
               self.loss_function
           );
           tree.fit(&item, &target);
           self.trees.push(tree);
           counter += 1; 
        }

        Ok(())
    }


    pub fn load_trees(
        &mut self,
        filepath: &str,
        features: &NDArray<f64>,
        target: &NDArray<f64>) {

        let paths = fs::read_dir(filepath).unwrap();
        let mut tree_count = 0;

        for path in paths {
            let path_str = path.unwrap().path().display().to_string();
            let mut dt =  DecisionTreeRegressor::load(
                &path_str,
                self.max_depth, 
                self.samples_split,
                self.loss_function
            ); 
            dt.fit(features, target);
            self.trees.push(dt);
            tree_count += 1; 
        }
        self.n_trees = tree_count;
    }


    pub fn fit(&mut self, features: &NDArray<f64>, target: &NDArray<f64>) {
        self.bootstrap_trees(features, target).unwrap();
    }


    pub fn fit_loaded(
        &mut self,
        features: &NDArray<f64>,
        target: &NDArray<f64>,
        filepath: &str) {

        self.load_trees(
            filepath,
            features,
            target
        );
    }


    pub fn predict(&self, features: NDArray<f64>) -> NDArray<f64> {
        let rows = features.shape().dim(0);
        let mut y_predictions: Vec<f64> = Vec::new(); 
        for item in 0..rows {
            let row = features.axis(0, item).unwrap(); 
            let mut predictions = Vec::new(); 
            for tree in self.trees() {
                let y_pred = tree.prediction(
                    row.clone(), tree.root().clone()
                );
                predictions.push(y_pred); 
            }

            let pred_val = NDArray::array(
                vec![predictions.len(), 1],
                predictions
            ).unwrap(); 
            let prediction = pred_val.avg(); 
            y_predictions.push(prediction);
        }

        NDArray::array(vec![rows, 1], y_predictions).unwrap() 
    }


    pub fn load(
        max_depth: usize,
        samples_split: usize,
        loss_function: fn(
            y_true: &NDArray<f64>, 
            y_pred: &NDArray<f64>) -> Result<f64, String>
        ) -> RandomForestRegressor {
          
        RandomForestRegressor {
            max_depth: max_depth,
            samples_split: samples_split,
            n_trees: 0,
            num_features: 0,
            loss_function: loss_function,
            trees: Vec::new()
        }
    }

}

