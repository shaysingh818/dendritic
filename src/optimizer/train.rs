use rand::thread_rng;
use rand::prelude::SliceRandom;
use indicatif::{ProgressBar, ProgressStyle}; 
use ndarray::{s, Axis};
use serde::Serialize;
use serde_json; 

use crate::optimizer::model::*;
use crate::optimizer::optimizers::Optimizer;
use crate::optimizer::regression::elastic::*; 
use crate::optimizer::regression::lasso::*; 
use crate::optimizer::regression::ridge::*; 
use crate::optimizer::regression::sgd::*; 
use crate::optimizer::regression::logistic::*;


/// Trainable model trait
pub trait Trainable {

    /// Train full dataset.
    ///
    /// # Arguments
    ///
    /// * `epochs` - Number of iterations to train the model.
    ///
    fn train(&mut self, epochs: usize);

    /// Train batches of data with random shuffling.
    ///
    /// # Arguments
    ///
    /// * `iterations` - The number of iterations to train the model.
    /// * `batch_size` - The size of each training batch.
    /// * `batch_epochs` - The number of epochs to train within each batch.
    ///
    fn train_batch(
        &mut self, 
        iterations: usize,
        batch_size: usize,
        batch_epochs: usize
    );

}


/// Trainable model trait with optimizer
pub trait TrainOptimizer {

    /// Train dataset with optimizer.
    ///
    /// # Arguments
    ///
    /// * `epochs` - The number of epochs to train on dataset.
    /// * `optimizer` - The optimizer to use for updating parameters on each iteration.
    ///
    fn train_with_optimizer<O: Optimizer>(
        &mut self, 
        epochs: usize, 
        optimizer: &mut O
    );

    
    /// Train batches of data with random shuffling & optimizer.
    ///
    /// # Arguments
    ///
    /// * `iterations` - The number of iterations to train the model.
    /// * `batch_size` - The size of each training batch.
    /// * `batch_epochs` - The number of epochs to train within each batch.
    /// * `optimizer` - The optimizer to use for updating parameters on each iteration.
    ///
    fn train_batch_with_optimizer<O: Optimizer>(
        &mut self, 
        iterations: usize,
        batch_size: usize,
        batch_epochs: usize,
        optimizer: &mut O
    );

}


#[derive(Serialize)]
struct TrainingResult {

    /// current loss of model during training
    loss: f64,

    /// Measure of how much loss decreased for each batch
    loss_decrease: f64, 

    /// Current batch number of training
    batch_number: String
}


macro_rules! train_default {

    ($t:ty) => {

        impl Trainable for $t {

            fn train(&mut self, epochs: usize) {

                let bar = ProgressBar::new(epochs.try_into().unwrap());
                    bar.set_style(ProgressStyle::default_bar()
                        .template("{bar:50} {pos}/{len}")
                        .unwrap());

                for _ in 0..epochs {
                    self.forward();
                    self.backward();
                    self.update_parameters();
                    bar.inc(1); 
                }

                bar.finish();

                let total_loss = self.loss();
                println!(
                    "Loss: {:?}, Epochs: {:?}", 
                    total_loss,
                    epochs
                );

            }

            
            fn train_batch(
                &mut self, 
                iterations: usize,
                batch_size: usize,
                batch_epochs: usize) {

                let x_train = self.input();
                let y_train = self.output(); 
                let rows = x_train.nrows();
                let num_batches = (rows + batch_size - 1) / batch_size;
                let mut curr_loss = 0.00;

                for iteration in 0..iterations {

                    let bar = ProgressBar::new(batch_epochs.try_into().unwrap());
                    bar.set_style(ProgressStyle::default_bar()
                        .template("{bar:50} {pos}/{len}")
                        .unwrap());
                
                    for _epoch in 0..batch_epochs {

                        let mut row_indices: Vec<_> = (0..rows).collect();
                        row_indices.shuffle(&mut thread_rng());

                        let x_shuffled = x_train.select(Axis(0), &row_indices);
                        let y_shuffled = y_train.select(Axis(0), &row_indices);


                        for batch_idx in 0..num_batches { 
                            let start_idx = batch_idx * batch_size;
                            let end_idx = (start_idx + batch_size).min(rows);
                            let x = x_shuffled.slice(s![start_idx..end_idx, ..]);
                            let y = y_shuffled.slice(s![start_idx..end_idx, ..]);

                            // fix this later
                            if (end_idx - start_idx) < batch_size {
                                continue; 
                            }

                            self.set_input(&x.to_owned());
                            self.set_output(&y.to_owned());

                            self.forward();
                            self.backward();

                            self.update_parameters();

                        }

                        bar.inc(1); 
                    }

                    bar.finish();



                    let total_loss = self.loss();
                    let result = TrainingResult {
                        loss: total_loss,
                        loss_decrease: curr_loss - total_loss,
                        batch_number: format!("{:?}/{:?}", iteration+1, iterations)
                    };
                    let json = serde_json::to_string_pretty(&result).unwrap();
                    println!("{}", json); 
                    curr_loss = total_loss; 
                    println!(""); 
                }




            }

        }

    }

}

train_default!(SGD);
train_default!(Logistic); 
train_default!(Ridge);
train_default!(Lasso); 
train_default!(Elastic); 


macro_rules! train_optimizer {

    ($t:ty) => {

        impl TrainOptimizer for $t {

            fn train_with_optimizer<O: Optimizer>(&mut self, epochs: usize, optimizer: &mut O) {

                let bar = ProgressBar::new(epochs.try_into().unwrap());
                    bar.set_style(ProgressStyle::default_bar()
                        .template("{bar:50} {pos}/{len}")
                        .unwrap());

                for _ in 0..epochs {
                    self.forward();
                    self.backward();
                    optimizer.step(self);
                    bar.inc(1); 
                }

                bar.finish();

                let total_loss = self.loss();
                println!(
                    "Loss: {:?}, Epochs: {:?}", 
                    total_loss,
                    epochs
                );

            }


            fn train_batch_with_optimizer<O: Optimizer>(
                &mut self, 
                iterations: usize,
                batch_size: usize,
                batch_epochs: usize,
                optimizer: &mut O) {

                let x_train = self.input();
                let y_train = self.output(); 
                let rows = x_train.nrows();
                let num_batches = (rows + batch_size - 1) / batch_size;
                let mut curr_loss = 0.00;

                for iteration in 0..iterations {

                    let bar = ProgressBar::new(batch_epochs.try_into().unwrap());
                    bar.set_style(ProgressStyle::default_bar()
                        .template("{bar:50} {pos}/{len}")
                        .unwrap());
                
                    for _epoch in 0..batch_epochs {

                        let mut row_indices: Vec<_> = (0..rows).collect();
                        row_indices.shuffle(&mut thread_rng());

                        let x_shuffled = x_train.select(Axis(0), &row_indices);
                        let y_shuffled = y_train.select(Axis(0), &row_indices);

                        for batch_idx in 0..num_batches { 
                            let start_idx = batch_idx * batch_size;
                            let end_idx = (start_idx + batch_size).min(rows);
                            let x = x_shuffled.slice(s![start_idx..end_idx, ..]);
                            let y = y_shuffled.slice(s![start_idx..end_idx, ..]);

                            // fix this later
                            if (end_idx - start_idx) < batch_size {
                                continue; 
                            }

                            self.set_input(&x.to_owned());
                            self.set_output(&y.to_owned());

                            self.forward();
                            self.backward();

                            optimizer.step(self);

                        }

                        bar.inc(1); 
                    }

                    bar.finish();

                    let total_loss = self.loss();
                    let result = TrainingResult {
                        loss: total_loss,
                        loss_decrease: curr_loss - total_loss,
                        batch_number: format!("{:?}/{:?}", iteration+1, iterations)
                    };
                    let json = serde_json::to_string_pretty(&result).unwrap();
                    println!("{}", json); 
                    curr_loss = total_loss; 
                    println!(""); 
                }


            }

        }

    }

}


train_optimizer!(SGD); 
