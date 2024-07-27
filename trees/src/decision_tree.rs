use ndarray::ndarray::NDArray;
use ndarray::ops::*;
use metrics::utils::*;
use crate::node::*;
use std::rc::Rc;


pub struct DecisionTreeClassifier {
    max_depth: usize, 
    samples_split: usize,
    root: Node
}


impl DecisionTreeClassifier {

    pub fn new(
        max_depth: usize,
        samples_split: usize) -> DecisionTreeClassifier {

        DecisionTreeClassifier {
            max_depth,
            samples_split,
            root: Node::leaf(0.0)
        }

    }

    pub fn build_tree(
        &self, 
        features: NDArray<f64>, 
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


            let (left, right) = self.split(
                features.clone(),
                threshold, 
                feature_idx
            );

            if info_gain > 0.0 {

                let left_subtree = self.build_tree(left, curr_depth+1); 
                let right_subtree = self.build_tree(right, curr_depth+1);

                return Node::new(
                   features.clone(),
                   threshold,
                   feature_idx,
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


    pub fn best_split(&self, features: NDArray<f64>) -> (f64, usize, f64) {

        let mut max_info_gain = f64::NEG_INFINITY;
        let mut feature_index = 0;
        let mut selected_threshold = f64::NEG_INFINITY;

        let num_features = features.shape().dim(1) - 1;
        for feat_idx in 0..num_features {
            let feature = features.axis(1, feat_idx).unwrap();
            let thresholds = feature.unique();
            for threshold in thresholds {

                let (left, right) = self.split(
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


    pub fn information_gain(
        &self,
        feature: NDArray<f64>,
        left: NDArray<f64>, 
        right: NDArray<f64>) -> f64 {

        let feature_entropy = entropy(feature.clone());
        let left_e = entropy(left.clone());
        let right_e = entropy(right.clone());

        let left_l = left.size() as f64 / feature.size() as f64;
        let right_l = right.size() as f64 / feature.size() as f64;
        let child_entropy = left_l * left_e + right_l * right_e;

        feature_entropy - child_entropy
    }

    pub fn split(
        &self,
        features: NDArray<f64>,
        threshold: f64,
        feature_idx: usize) -> (NDArray<f64>, NDArray<f64>) {

        let mut idx_counter = 0;
        let mut left_indices: Vec<usize> = Vec::new();
        let mut right_indices: Vec<usize> = Vec::new();

        let feature = features.axis(1, feature_idx).unwrap();

        for feat in feature.values() {

            if *feat > threshold {
                right_indices.push(idx_counter);
            }

            if *feat <= threshold {
                left_indices.push(idx_counter);
            }

            idx_counter += 1;
        }

        let left = features.axis_indices(0, left_indices).unwrap();
        let right = features.axis_indices(0, right_indices).unwrap();
        (left, right)

    }

    pub fn select_max_class(&self, target: NDArray<f64>) -> f64 {
        let max = target.values().iter().max_by(
            |a, b| a.total_cmp(b)
        ).unwrap();
        *max
    }

    pub fn fit(&mut self, features: NDArray<f64>, target: NDArray<f64>) {
       self.root = self.build_tree(features, 0);
       self.print_tree(self.root.clone(), 2); 
    }

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

    pub fn print_tree(&self, node: Node, level: usize) {
    
        let right = node.right();
        let left = node.left();

        if node.value().is_some() {
            println!("{:?}", node.value().unwrap());
        } else {
            println!("{:?} <= {:?}", node.feature_idx(), node.threshold());

            print!("{:ident$}left: ", "", ident=level); 
            match left {
                Some(left) => self.print_tree(left, level+2),
                None => println!("")
            }

            print!("{:ident$}right: ", "", ident=level); 
            match right {
                Some(right) => self.print_tree(right, level+2),
                None => println!("")
            }   
        }

    }

}
