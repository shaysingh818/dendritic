use crate::regression::*; 


pub trait RegressionTrain {

    fn train(&mut self, epochs: usize);

    fn train_batch(&mut self, epochs: usize, batch_size: usize);

}


impl RegressionTrain for Regression {
    
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

        let x_train = self.graph.node(0).output();
        let y_train = self.graph.node(5).output(); 
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


}


macro_rules! impl_regression_train {

    ($t:ty) => {

        impl RegressionTrain for $t {

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

                let x_train = self.regression.graph.node(0).output();
                let y_train = self.regression.graph.node(5).output(); 
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


        }

    };

}

impl_regression_train!(Lasso); 
impl_regression_train!(Ridge); 
