use std::rc::Rc;
use std::fs::{File}; 
use std::io::{Read};
use std::cell::{RefCell};

use ndarray::ndarray::NDArray;
use crate::node::*;


/// Split dataset into two parts using threhsold values
pub fn split(
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


/// Save entire instance of tree using recursion
pub fn save_tree(node: Node, node_save: &mut NodeSerialized) {

    let right = node.right();
    let left = node.left();

    match left {
        Some(left) => {
            let mut left_save = left.save();
            save_tree(left, &mut left_save);
            node_save.left = Some(Box::new(left_save)); 

        },
        None => {
            return;
        }
    }

    match right {
        Some(right) => {
            let mut right_save = right.save();
            save_tree(right, &mut right_save);
            node_save.right = Some(Box::new(right_save));

        },
        None => {
            return;
        }
    }

}


/// Load entire instance of tree using recursion
pub fn load_tree(node: &mut Node, node_save: NodeSerialized) {

    let left = node_save.left; 
    let right = node_save.right;

    match left {
        Some(left) => {
            let mut left_ptr = Node::load(*left.clone()); 
            load_tree(&mut left_ptr, *left);
            node.left = Some(Rc::new(RefCell::new(left_ptr))); 

        },
        None => {
            return;
        }
    }

    match right {
        Some(right) => {
            let mut right_ptr = Node::load(*right.clone()); 
            load_tree(&mut right_ptr, *right);
            node.right = Some(Rc::new(RefCell::new(right_ptr))); 
        },
        None => {
            return;
        }
    }

}


/// Load Instance of saved NodeSerialized structure
pub fn load_root(filepath: &str) -> std::io::Result<NodeSerialized> {
    let filename_format = format!("{}/tree.json", filepath);
    let mut file = match File::open(filename_format) {
        Ok(file) => file,
        Err(err) => {
            return Err(err);
        }
    };
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let instance: NodeSerialized = serde_json::from_str(&contents)?;
    Ok(instance)
}


/// Print decision tree that has atleast two children
pub fn print_tree(node: Node, level: usize) {

    let right = node.right();
    let left = node.left();

    if node.value().is_some() {
        println!("{:?}", node.value().unwrap());
    } else {
        println!("{:?} <= {:?}", node.feature_idx(), node.threshold());

        print!("{:ident$}left: ", "", ident=level); 
        match left {
            Some(left) => print_tree(left, level+2),
            None => println!("")
        }

        print!("{:ident$}right: ", "", ident=level); 
        match right {
            Some(right) => print_tree(right, level+2),
            None => println!("")
        }   
    }

}

