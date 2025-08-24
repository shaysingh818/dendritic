use std::fs;
use std::fs::File;


use ndarray::{arr2, Array2};

use dendritic::optimizer::model::*;
use dendritic::optimizer::train::*;
use dendritic::optimizer::regression::logistic::*;


fn load_multi_class() -> (Array2<f64>, Array2<f64>) {

	// multi class data
	let x = arr2(&[	
		[1.0, 2.0],
		[1.5, 1.8],
		[2.0, 1.0], // Class 0
		[4.0, 4.5],
		[4.5, 4.8],
		[5.0, 5.2], // Class 1
		[7.0, 7.5],
		[7.5, 8.0],
		[8.0, 8.5], // Class 2
	]);

	// Label encoded target values
	let y = arr2(&[
		[1.0, 0.0, 0.0],
		[1.0, 0.0, 0.0],
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, 1.0],
		[0.0, 0.0, 1.0],
		[0.0, 0.0, 1.0],
	]);
	
	(x, y)
}

// sample testing data
fn load_binary_data() -> (Array2<f64>, Array2<f64>) {

 	// binary logstic data
	let x = arr2(&[
		[1.0, 2.0],
		[2.0, 1.0],
		[1.5, 1.8],
		[3.0, 3.2],
		[2.8, 3.0],
		[5.0, 5.5],
		[6.0, 5.8],
		[5.5, 6.0],
		[6.2, 5.9],
		[7.0, 6.5]
	]);

	let y = arr2(&[
		[0.0],
		[0.0],
		[0.0],
		[0.0],
		[0.0],
		[1.0],
		[1.0],
		[1.0],
		[1.0],
		[1.0]
	]);
	(x, y)
}


fn main() -> std::io::Result<()> {

	// Multi class logistic model
	let (x, y) = load_binary_data();
	let mut model = Logistic::new(&x, &y, false, 0.01).unwrap();

	// Multi class logistic model
	let (x1, y1) = load_multi_class();
	let mut multi_class_model = Logistic::new(&x1, &y1, true, 0.01).unwrap();

	// train and save logistic model
	model.train(1000);
	model.save("data/logistic");

	// train and save multi class model using same methods
	multi_class_model.train(2000);
	multi_class_model.save("data/multiclass_logistic")?;

	// load results of multi class model and make predictions
	let mut loaded = Logistic::load("data/multiclass_logistic").unwrap();
	let output = loaded.predict(&x1);

    // output.column(0) gives the class prediction, column(1) gives the probability
    println!("Class Predictions: {:?}", output);
	Ok(())
}