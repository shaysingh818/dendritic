use std::fs; 
use std::fs::{File}; 
use std::io::{BufWriter, Write};

use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;
use crate::node::*;
use crate::utils::*; 


pub struct DecisionTreeClassifier {
    max_depth: usize, 
    samples_split: usize,
    metric_function: fn(x: NDArray<f64>) -> f64,
    root: Node
}


impl DecisionTreeClassifier {

    /// Create instance of decision tree classifier
    pub fn new(
        max_depth: usize,
        samples_split: usize,
        metric_function: fn(x: NDArray<f64>) -> f64
    ) -> DecisionTreeClassifier {

        DecisionTreeClassifier {
            max_depth,
            samples_split,
            metric_function: metric_function,
            root: Node::leaf(0.0)
        }

    }

    /// Return root node of decision tree classifier tree
    pub fn root(&self) -> &Node {
        &self.root
    }


    /// Build decision tree classifier
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
                info_gain, 
                feature_idx, 
                threshold
            ) = self.best_split(features.clone());


            let (left, right) = split(
                features.clone(),
                threshold, 
                feature_idx
            );

            if info_gain > 0.0 {

                let left_subtree = self.build_tree(&left, curr_depth+1); 
                let right_subtree = self.build_tree(&right, curr_depth+1);

                return Node::new(
                   features.clone(),
                   threshold,
                   feature_idx,
                   info_gain,
                   left_subtree,
                   right_subtree
                );
            }
        }


        let leaf_val = self.select_max_class(
            features.axis(1, num_features-1
        ).unwrap());

        Node::leaf(leaf_val)
    }   


    /// Find optimal split for decision tree classifier
    pub fn best_split(&self, features: NDArray<f64>) -> (f64, usize, f64) {

        let mut max_info_gain = f64::NEG_INFINITY;
        let mut feature_index = 0;
        let mut selected_threshold = f64::NEG_INFINITY;

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

                    let info_gain = self.information_gain(
                        features.axis(1, num_features).unwrap(), 
                        left.axis(1, num_features).unwrap(), 
                        right.axis(1, num_features).unwrap()
                    );

                    if info_gain > max_info_gain {
                        max_info_gain = info_gain;
                        feature_index = feat_idx;
                        selected_threshold = threshold;
                    }
                }

            }
        }

        (max_info_gain, feature_index, selected_threshold)
    }

    /// Calculate information gain for decision tree classifier
    pub fn information_gain(
        &self,
        feature: NDArray<f64>,
        left: NDArray<f64>, 
        right: NDArray<f64>) -> f64 {

        let feature_entropy = (self.metric_function)(feature.clone());
        let left_e = (self.metric_function)(left.clone());
        let right_e = (self.metric_function)(right.clone());

        let left_l = left.size() as f64 / feature.size() as f64;
        let right_l = right.size() as f64 / feature.size() as f64;
        let child_entropy = left_l * left_e + right_l * right_e;

        feature_entropy - child_entropy
    }


    /// Select highest probability class from target output
    pub fn select_max_class(&self, target: NDArray<f64>) -> f64 {
        let max = target.values().iter().max_by(
            |a, b| a.total_cmp(b)
        ).unwrap();
        *max
    }

    /// Fit decision tree classifier to features and target
    pub fn fit(
        &mut self, 
        features: &NDArray<f64>, 
        _target: &NDArray<f64>) {
       self.root = self.build_tree(features, 0);
    }

    /// Generate prediction for row sample of decision tree classifier
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

    /// Generate prediction for all row samples for decision tree classification
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


    /// Save model parameters for decision tree classification
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


    /// Load model parameters for decision tree classification
    pub fn load(
        filepath: &str, 
        max_depth: usize,
        samples_split: usize,
        metric_function: fn(x: NDArray<f64>) -> f64) -> DecisionTreeClassifier {

        let root = load_root(filepath).unwrap();
        let root_node = Node::load(root.clone());
        load_tree(&mut root_node.clone(), root);  

        DecisionTreeClassifier {
            max_depth,
            samples_split,
            metric_function: metric_function,
            root: root_node
        }
 
    }


}




