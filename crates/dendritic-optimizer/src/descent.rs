

pub trait DescentOptimizer {

    /// Definition of differentiable function using computation graph
    fn function_definition(&mut self); 

    /// Implementation for how each epoch updates in the forward and backward pass
    fn parameter_update(&mut self); 

    /// Implementation of training cycle for full dataset
    fn train(&mut self, epochs: usize);

    /// Training cycle for mini batches or partial fitting
    fn train_batch(&mut self, epochs: usize, batch_size: usize);

    /// Save single instance of parameters to file path
    fn save(&self, filepath: &str) -> std::io::Result<()>; 

    /// Save snapshot of parameters to a namespace
    fn save_snapshot(&self, namespace: &str) -> std::io::Result<()>; 

    /// Load single instance of parameters from file path
    fn load(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> where Self: Sized; 

    /// Load snapshot of parameters from a namespace
    fn load_snapshot(
        namespace: &str, 
        year: &str, 
        month: &str, 
        day: &str,
        snapshot_id: &str) -> Result<Self, Box<dyn std::error::Error>> where Self: Sized; 

}

