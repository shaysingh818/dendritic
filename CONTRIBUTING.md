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


fn main() -> std::io::Result<()> {

	// Sample dataset
	let x = arr2(&[
		[1.0, 2.0, 3.0],
		[2.0, 3.0, 4.0],
		[3.0, 4.0, 5.0],
		[4.0, 5.0, 6.0],
		[5.0, 6.0, 7.0]
	]);

	let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);

	// Ridge regression model with lambda parameters
	let mut model = let mut model = Ridge::new(&x, &y, 0.001, 0.001).unwrap();
	
	// train & save for later user
	model.train(1000);
	model.save("data/ridge")?;
	
	// Load saved model to make predictions
	let mut loaded_model = Ridge::load("data/ridge").unwrap();
	let output = loaded_model.predict(&x);
	let diff = output - y;
	Ok(())
}
```