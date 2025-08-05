use std::io::Write; 

use chrono::Local; 
use ndarray::{arr2, Array2, Axis};

use dendritic_autodiff::operations::activation::*; 
use dendritic_autodiff::operations::loss::*; 
use dendritic_optimizer::regression::logistic::*; 
use dendritic_optimizer::regression::sgd::*;
use dendritic_optimizer::train::*;
use dendritic_optimizer::model::*;
use dendritic_optimizer::optimizers::*; 
use dendritic_optimizer::optimizers::Optimizer;

pub fn load_data() -> (Array2<f64>, Array2<f64>) {

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

pub fn load_multi_class() -> (Array2<f64>, Array2<f64>) {

    let x1 = arr2(&[
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 1.0],   // Class 0
        [4.0, 4.5],
        [4.5, 4.8],
        [5.0, 5.2],   // Class 1
        [7.0, 7.5],
        [7.5, 8.0],
        [8.0, 8.5],   // Class 2
    ]);

    let y1 = arr2(&[
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

    (x1, y1)
}

pub fn load_binary_data() -> (Array2<f64>, Array2<f64>) {

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



pub fn nesterov() {

    // nesterov momentum
    let lr = 0.001; // learning rate
    let B = 0.9; // momentum factor
    let mut v_w = Array2::zeros((3, 1)); // velocity for weights
    let mut v_b = Array2::zeros((1,1)); // velocity for biases
    let (x, y) = load_data(); 
    
    let mut model = SGD::new(&x, &y, lr).unwrap();
    
    for _ in 0..250 {

        model.graph.forward(); 
        model.graph.backward();

        let mut w = model.graph.node(1); 
        let mut b = model.graph.node(3); 

        // factor in velocity
        let w_lookahead = w.output() - (B * v_w.clone()); 
        let b_lookahead = b.output() - (B * v_b.clone()); 

        model.graph.mut_node_output(1, w_lookahead);
        model.graph.mut_node_output(3, b_lookahead);

        let w_grad = w.grad() * lr; 
        let b_grad = b.grad() * lr;

        v_w = w_grad + (B * v_w); 
        v_b = b_grad + (B * v_b);

        let new_w = w.output() - v_w.clone(); 
        let new_b = b.output() - v_b.clone();

        model.graph.mut_node_output(1, new_w); 
        model.graph.mut_node_output(3, new_b);

        let loss_total = model.loss();
        println!(
            "\nLoss: {:?}", 
            loss_total
        );

    }

    println!("{:?}", model.predicted()); 

}


pub fn adagrad() {

    let lr = 0.9; // learning rate
    let epsilon = 1e-8; // momentum factor
    let (x, y) = load_data();
    let mut s_w: Array2<f64> = Array2::zeros((3, 1)); // velocity for weights
    let mut s_b: Array2<f64> = Array2::zeros((1,1)); // velocity for biases
    let mut model = SGD::new(&x, &y, lr).unwrap();
    
    for _ in 0..200 {

        model.graph.forward(); 
        model.graph.backward();

        let mut w = model.graph.node(1); 
        let mut b = model.graph.node(3);
        let w_grad = w.grad();
        let b_grad = b.grad();

        // square the params
        let w_grad_squared = w_grad.mapv(|x| x * x); 
        let b_grad_squared = b_grad.mapv(|x| x * x); 

        s_w += &w_grad_squared;
        s_b += &b_grad_squared;

        let w_ada = lr / (s_w.mapv(f64::sqrt) + epsilon); 
        let b_ada = lr / (s_b.mapv(f64::sqrt) + epsilon);

        let w_new = w.output() - (w_ada * w.grad());
        let b_new = b.output() - (b_ada * &b_grad);

        model.graph.mut_node_output(1, w_new); 
        model.graph.mut_node_output(3, b_new);

        let loss_total = model.loss();
        println!(
            "\nLoss: {:?}", 
            loss_total
        );

    }

    println!("{:?}", model.predicted()); 

}

pub fn rmsprop() {

    let lr = 0.1; // learning rate
    let epsilon = 1e-8; // squared gradients
    let decay_rate = 0.9;
    let (x, y) = load_data();
    let mut s_w: Array2<f64> = Array2::zeros((3, 1)); // velocity for weights
    let mut s_b: Array2<f64> = Array2::zeros((1,1)); // velocity for biases
    let mut model = SGD::new(&x, &y, lr).unwrap();
    
    for _ in 0..500 {

        model.graph.forward(); 
        model.graph.backward();

        let mut w = model.graph.node(1); 
        let mut b = model.graph.node(3);

        let w_grad = w.grad();
        let b_grad = b.grad(); 

        // square the params
        let w_grad_squared = w_grad.mapv(|x| x * x); 
        let b_grad_squared = b_grad.mapv(|x| x * x);

        s_w = decay_rate * s_w + (1.0 - decay_rate) * w_grad_squared;
        s_b = decay_rate * s_b + (1.0 - decay_rate) * b_grad_squared;

        let w_rms = lr / (s_w.mapv(f64::sqrt) + epsilon); 
        let b_rms = lr / (s_b.mapv(f64::sqrt) + epsilon);

        let w_new = w.output() - (w_rms * w_grad);
        let b_new = b.output() - (b_rms * &b_grad);

        model.graph.mut_node_output(1, w_new); 
        model.graph.mut_node_output(3, b_new);

        let loss_total = model.loss();
        println!(
            "\nLoss: {:?}", 
            loss_total
        );
        
    }

    println!("{:?}", model.predicted()); 

}


pub fn adadelta() {

    let ys = 0.95; // gradient decay
    let yx = 0.95; // update decay
    let epsilon = 1e-6; // squared gradients
    let (x, y) = load_data();

    let mut s_w: Array2<f64> = Array2::zeros((3, 1)); // sum of squared grad for weights
    let mut s_b: Array2<f64> = Array2::zeros((1,1)); // sum of squared grad for biases 

    let mut u_w: Array2<f64> = Array2::zeros((3, 1)); // sum of squared grad for weights
    let mut u_b: Array2<f64> = Array2::zeros((1,1)); // sum of squared grad for biases

    let mut model = SGD::new(&x, &y, 0.01).unwrap();
    
    for _ in 0..1000 {

        model.graph.forward(); 
        model.graph.backward();

        let mut w = model.graph.node(1); 
        let mut b = model.graph.node(3);

        let w_grad = w.grad();
        let b_grad = b.grad(); 

        // square the params
        let w_grad_squared = w_grad.mapv(|x| x * x); 
        let b_grad_squared = b_grad.mapv(|x| x * x);

        s_w = ys * s_w + (1.0 - ys) * w_grad_squared;
        s_b = ys * s_b + (1.0 - ys) * b_grad_squared;

        let w_delta = ((u_w.mapv(f64::sqrt) + epsilon) / (s_w.mapv(f64::sqrt) + epsilon)) * w_grad; 
        let b_delta = ((u_b.mapv(f64::sqrt) + epsilon) / (s_b.mapv(f64::sqrt) + epsilon)) * b_grad; 

        u_w = yx * u_w + (1.0 - yx) * w_delta.mapv(|x| x * x);
        u_b = yx * u_b + (1.0 - yx) * b_delta.mapv(|x| x * x);

        let w_new = w.output() + (w_delta * -1.0);
        let b_new = b.output() + (b_delta * -1.0);

        model.graph.mut_node_output(1, w_new); 
        model.graph.mut_node_output(3, b_new);

        let loss_total = model.loss();
        println!(
            "\nLoss: {:?}", 
            loss_total
        );
        
    }

}

pub fn adam() {

    let lr = 0.1; 
    let yv = 0.9; // gradient decay
    let ys = 0.999; // update decay
    let epsilon = 1e-6; // squared gradients
    let (x, y) = load_data();


    let mut k = 0;
    let mut s_w: Array2<f64> = Array2::zeros((3, 1)); // sum of squared grad for weights
    let mut s_b: Array2<f64> = Array2::zeros((1,1)); // sum of squared grad for biases 

    let mut v_w: Array2<f64> = Array2::zeros((3, 1)); // sum of squared grad for weights
    let mut v_b: Array2<f64> = Array2::zeros((1,1)); // sum of squared grad for biases

    let mut model = SGD::new(&x, &y, 0.01).unwrap();
    
    for _ in 0..500 {

        model.graph.forward(); 
        model.graph.backward();

        let mut w = model.graph.node(1); 
        let mut b = model.graph.node(3);

        let w_grad = w.grad();
        let b_grad = b.grad(); 

        let w_grad_squared = w_grad.mapv(|x| x * x); 
        let b_grad_squared = b_grad.mapv(|x| x * x);

        v_w = yv * v_w + (1.0 - yv) * w_grad;
        v_b = yv * v_b + (1.0 - yv) * b_grad;

        s_w = ys * s_w + (1.0 - ys) * w_grad_squared;
        s_b = ys * s_b + (1.0 - ys) * b_grad_squared;
        k += 1; 

        let v_w_hat = v_w.clone() / (1.0 - yv.powf(k as f64));
        let v_b_hat = v_b.clone() / (1.0 - yv.powf(k as f64)); 

        let s_w_hat = s_w.clone() / (1.0 - ys.powf(k as f64));
        let s_b_hat = s_b.clone() / (1.0 - ys.powf(k as f64));

        let w_delta = lr * v_w_hat / (s_w_hat.mapv(f64::sqrt) + epsilon); 
        let b_delta = lr * v_b_hat / (s_b_hat.mapv(f64::sqrt) + epsilon); 

        model.graph.mut_node_output(1, w.output() - w_delta); 
        model.graph.mut_node_output(3, b.output() - b_delta);

        let loss_total = model.loss();
        println!(
            "\nLoss: {:?}", 
            loss_total
        );
        
    }


    println!("{:?}", model.predicted()); 

}

fn main() -> std::io::Result<()> {

    let (x, y) = load_data();
    let mut model = SGD::new(&x, &y, 0.1).unwrap();
    let mut optimizer = RMSProp::default(&model);

    //model.train_v1(250, Some(&mut optimizer));

    /*
    for _ in 0..250 {
        model.graph.forward(); 
        model.graph.backward();
        optimizer.step(&mut model);

        let loss_total = model.loss();
        println!(
            "\nLoss: {:?}", 
            loss_total
        );
    } */ 

    Ok(())

}
