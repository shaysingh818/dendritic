use rand::thread_rng;
use rand::prelude::SliceRandom;
use uuid::Uuid;
use chrono::{Datelike, Utc};  
use indicatif::{ProgressBar, ProgressStyle}; 
use ndarray::{s, Array2, Axis}; 

use crate::regression::*; 
use crate::classification::*; 

pub trait Trainable {

    fn train(&mut self, epochs: usize);

    fn train_batch(&mut self, epochs: usize, batch_size: usize);

    fn predict(&mut self, x: &Array2<f64>) -> Array2<f64>; 

}

pub trait RegressionFit {

    fn fit(&mut self, epochs: usize);

    fn fit_batch(
        &mut self, 
        x_train: Array2<f64>, 
        y_train: Array2<f64>,
        batch_size: usize,
        num_batches: usize,
        rows: usize);

}

macro_rules! impl_fit {

    ($t:ty) => {

        impl RegressionFit for $t {

            fn fit(&mut self, epochs: usize) {

                let bar = ProgressBar::new(epochs.try_into().unwrap());
                bar.set_style(ProgressStyle::default_bar()
                    .template("{bar:50} {pos}/{len}")
                    .unwrap());

                for _ in 0..epochs {
                    self.graph.forward();
                    self.graph.backward(); 
                    self.parameter_update();
                    bar.inc(1); 
                }

                bar.finish();
            }

            fn fit_batch(
                &mut self, 
                x_train: Array2<f64>, 
                y_train: Array2<f64>,
                batch_size: usize,
                num_batches: usize,
                rows: usize) {

                let bar = ProgressBar::new(1000);
                bar.set_style(ProgressStyle::default_bar()
                    .template("{bar:50} {pos}/{len}")
                    .unwrap());

                for _ in 0..1000 {

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

                        self.graph.forward();
                        self.graph.backward(); 
                        self.parameter_update();
                    }
                    bar.inc(1); 
                }

                bar.finish(); 
            }

        }

    }

}

impl_fit!(Regression);
impl_fit!(Logistic);


macro_rules! train {

    ($t:ty) => {

        impl Trainable for $t {

            fn train(&mut self, epochs: usize) {

                self.fit(epochs);
                let total_loss = self.measure_loss();
                println!(
                    "Loss: {:?}, Learning Rate: {:?}", 
                    total_loss,
                    self.learning_rate
                );                
            }

            fn train_batch(&mut self, epochs: usize, batch_size: usize) {

                if epochs % 1000 != 0 {
                    panic!("Number of epochs must be evenly divisible by 1000");
                }

                let x_train = self.input();
                let y_train = self.output(); 
                let rows = x_train.nrows();
                let num_batches = (rows + batch_size - 1) / batch_size;

                let epoch_batches = epochs / 1000;
                for _ in 0..epoch_batches {

                    self.fit_batch(
                        x_train.clone(), 
                        y_train.clone(), 
                        batch_size, 
                        num_batches, 
                        rows
                    );

                    let loss_total = self.measure_loss();
                    println!(
                        "\nLoss: {:?}, Learning Rate: {:?}, Epochs: {:?}", 
                        loss_total,
                        self.learning_rate,
                        epochs
                    );
                    println!(""); 
                }

            }

            fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {

                let y = Array2::zeros((x.nrows(), self.output().shape()[1]));
                self.set_input(x);
                self.set_output(&y);

                self.graph.forward();
                self.graph.backward(); 
                self.predicted()
            }

        }

    }

}

train!(Regression); 
train!(Logistic);


macro_rules! impl_regression_extension_train {

    ($t:ty) => {

        impl Trainable for $t {

            fn train(&mut self, epochs: usize) {

                self.regression.fit(epochs);
                let total_loss = self.measure_loss();
                println!(
                    "Loss: {:?}, Learning Rate: {:?}", 
                    total_loss,
                    self.regression.learning_rate
                );
                
            }

            fn train_batch(&mut self, epochs: usize, batch_size: usize) {

                if epochs % 1000 != 0 {
                    panic!("Number of epochs must be evenly divisible by 1000");
                }

                let x_train = self.input();
                let y_train = self.output(); 
                let rows = x_train.nrows();
                let num_batches = (rows + batch_size - 1) / batch_size;

                let epoch_batches = epochs / 1000;
                for _ in 0..epoch_batches {

                    self.regression.fit_batch(
                        x_train.clone(), 
                        y_train.clone(), 
                        batch_size, 
                        num_batches, 
                        rows
                    );

                    let loss_total = self.measure_loss();
                    println!(
                        "\nLoss: {:?}, Learning Rate: {:?}, Epochs: {:?}", 
                        loss_total,
                        self.regression.learning_rate,
                        epochs
                    );
                    println!(""); 
                }

            }

            fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
                self.regression.predict(&x.to_owned())
            }

        }

    };

}

impl_regression_extension_train!(Lasso); 
impl_regression_extension_train!(Ridge);
impl_regression_extension_train!(Elastic);


