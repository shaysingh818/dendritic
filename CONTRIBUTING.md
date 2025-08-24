# Contributing to Dendritic 🧠📐

First off, thanks for taking the time to contribute! Dendritic is a lightweight and extensible optimization library focused on clean abstractions for machine learning and mathematical optimization. Your help makes it better for everyone.

Whether you're here to file a bug, request a feature, or contribute code/documentation, you're in the right place!

---

## 🛠️ How to Contribute

### 1. **Discuss Before You Code**

Open an issue before submitting a PR, especially for:
- New features (e.g., optimizers, preprocessing utilities)
- API/trait changes
- Performance optimizations
- Refactoring major components
### 2. **Fork and Branch**

- Fork the repository.
- Create a branch based on the issue/topic. Use descriptive names like `feature/second-order-optim`, `bugfix/sgd-overflow`, etc.
### 3. **Code Style**

- Use idiomatic Rust.
- Document public functions, types, and modules.
- Prefer composable abstractions; keep the library extensible.
### 4. **Testing**

- Add tests for any new logic.
- Run existing tests with:

```bash
cargo test
```



```rust
use ndarray::{arr2, Array2};

use dendritic::optimizer::model::*;
use dendritic::optimizer::optimizers::*;
use dendritic::optimizer::optimizers::Optimizer;
use dendritic::optimizer::regression::sgd::*;


// sample testing data
fn load_sample_data() -> (Array2<f64>, Array2<f64>) {

 	let x = arr2(&[
		[1.0, 2.0, 3.0],
		[2.0, 3.0, 4.0],
		[3.0, 4.0, 5.0],
		[4.0, 5.0, 6.0],
		[5.0, 6.0, 7.0]
	]);
	
	let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);
	(x, y)
}


fn main() {

	let alpha = 0.1; // adjust learning rate to see different results
	
	// Load data and set optimizers
	let (x, y) = load_sample_data();
	let mut model = SGD::new(&x, &y, alpha).unwrap();
	let mut optimizer = Adam::default(&model);
	
	// Train
	model.train(1000);
		  
	// Retrieve loss and predicted samples
	let loss_total = model.loss();
	let predicted = model.predicted().mapv(|x| x.round());
	println!("Loss: {:?}", loss_total);
	println!("Predictions: {:?}", model.predicted());
}

```

```rust
use std::fs;
use std::fs::File;


use ndarray::{arr2, Array2};
use dendritic::optimizer::model::*;
use dendritic::optimizer::train::*
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
	model.save("data/logistic")

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
```