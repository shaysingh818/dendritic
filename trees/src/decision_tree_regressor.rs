use std::fs; 
use std::fs::{File}; 
use std::io::{BufWriter, Write};

use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use crate::node::*;
use crate::utils::*; 


pub struct DecisionTreeRegressor {
    max_depth: usize, 
    samples_split: usize,
    root: Node,
    loss_function: fn(
        y_true: &NDArray<f64>, 
        y_pred: &NDArray<f64>) -> Result<f64, String>
}


impl DecisionTreeRegressor {


    /// Create new instance of decision tree regression model
    pub fn new(
        max_depth: usize,
        samples_split: usize,
        loss_function: fn(
            y_true: &NDArray<f64>, 
            y_pred: &NDArray<f64>) -> Result<f64, String>
    ) -> DecisionTreeRegressor {

        DecisionTreeRegressor {
            max_depth,
            samples_split,
            root: Node::leaf(0.0),
            loss_function: loss_function,
        }

    }

    /// Retrieve root node of regression tree
    pub fn root(&self) -> &Node {
        &self.root
    }

    /// Build regression tree from root node
    pub fn build_tree(
        &self,
        features: &NDArray<f64>,
        curr_depth: usize) -> Node {

        let num_features = features.shape().dim(1);
        let num_samples = features.shape().dim(0);


        let depth_condition = curr_depth <= self.max_depth;
        let sample_condition = num_samples >= self.samples_split;

        if sample_condition && depth_condition {

            let (
                mse, 
                feature_idx, 
                threshold
            ) = self.best_split(features.clone());

            let (left, right) = split(
                features.clone(),
                threshold, 
                feature_idx
            );

            if mse > 0.0 {

                let left_subtree = self.build_tree(&left, curr_depth+1); 
                let right_subtree = self.build_tree(&right, curr_depth+1);

                return Node::regression(
                   features.clone(),
                   threshold,
                   feature_idx,
                   mse,
                   left_subtree,
                   right_subtree
                );
            }
        }

        let y_vals = features.axis(1, num_features-1).unwrap();
        let leaf_val = y_vals.avg();
        Node::leaf(leaf_val)
    }


    /// Find optimal split for regression tree
    pub fn best_split(&self, features: NDArray<f64>) -> (f64, usize, f64) {

        let mut feature_index = 0;
        let mut min_mse = f64::INFINITY;
        let mut selected_threshold = 0.0;

        let num_features = features.shape().dim(1) - 1;
        for feat_idx in 0..num_features {
            let feature = features.axis(1, feat_idx).unwrap();
            let thresholds = feature.unique();
            for threshold in thresholds {
 
                let (left, right) = split(
                    features.clone(),
                    threshold, 
                    feat_idx
                );

                if left.size() > 0 && right.size() > 0 {

                    let curr_mse = self.gain(
                        left.axis(1, num_features).unwrap(), 
                        right.axis(1, num_features).unwrap()
                    ); 

                    if curr_mse < min_mse {
                        min_mse = curr_mse; 
                        feature_index = feat_idx; 
                        selected_threshold = threshold; 
                    }
                }

            }
            
        }

        (min_mse, feature_index, selected_threshold)

    }

    /// Find information gain for regression tree
    pub fn gain(&self, left: NDArray<f64>, right: NDArray<f64>) -> f64 {

        let left_avg = NDArray::fill(
            left.shape().values(),
            left.avg()
        ).unwrap();

        let right_avg = NDArray::fill(
            right.shape().values(),
            right.avg()
        ).unwrap();

        let left_mse = (self.loss_function)(&left, &left_avg).unwrap();
        let right_mse = (self.loss_function)(&right, &right_avg).unwrap();
        let mse_total = left_mse + right_mse;
        mse_total
    }


    /// Generate row prediction for regression tree
    pub fn prediction(&self, inputs: NDArray<f64>, node: Node) -> f64 {

        let right = node.right();
        let left = node.left();

        if node.value().is_some() {
            return node.value().unwrap();
        }

        let feature_val = inputs.idx(node.feature_idx()); 
        if *feature_val <= node.threshold() {

            match left {
                Some(left) => self.prediction(inputs, left),
                None => -1.0
            }

        } else {

            match right {
                Some(right) => self.prediction(inputs, right),
None => -1.0
            }
        }

    }

    /// Predict outcomes for regression tree with all row samples
    pub fn predict(&self, input: NDArray<f64>) -> NDArray<f64> {
        let rows = input.shape().dim(0);
        let mut results = Vec::new();
        for item in 0..rows {
            let row = input.axis(0, item).unwrap();
            let val = self.prediction(row, self.root.clone());
            results.push(val);
        }

        NDArray::array(vec![rows, 1], results).unwrap()
    }


    /// Fit features and target for regression tree
    pub fn fit(&mut self, features: &NDArray<f64>, _target: &NDArray<f64>) {
       self.root = self.build_tree(features, 0);
    }


    /// Save model parameters for regression tree
    pub fn save(&self, filepath: &str) -> std::io::Result<()> {

        let mut node_save = self.root.save();
        save_tree(self.root.clone(), &mut node_save);

        let tree_path = format!("{}/tree.json", filepath);
        fs::create_dir_all(filepath)?;
 
        let file = match File::create(tree_path) {
            Ok(file) => file,
            Err(err) => {
                return Err(err);
            }
        };
        let mut writer = BufWriter::new(file);
        let json_string = serde_json::to_string_pretty(&node_save)?;
        writer.write_all(json_string.as_bytes())?;  
        Ok(())
    }


    /// Load model parameters for regression tree
    pub fn load(
        filepath: &str, 
        max_depth: usize,
        samples_split: usize,
        loss_function: fn(
            y_true: &NDArray<f64>, 
            y_pred: &NDArray<f64>) -> Result<f64, String>
        ) -> DecisionTreeRegressor {

        let root = load_root(filepath).unwrap();
        let root_node = Node::load(root.clone());
        load_tree(&mut root_node.clone(), root);  

        DecisionTreeRegressor {
            max_depth,
            samples_split,
            loss_function: loss_function,
            root: root_node
        }
 
    }




}
