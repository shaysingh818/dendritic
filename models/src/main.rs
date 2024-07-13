use datasets::iris::*;
use datasets::breast_cancer::*;
use datasets::diabetes::*;
use regression::logistic::*;
use metrics::loss::*;
use metrics::activations::*;


fn diabetes_model() {

    // load data
    let (x_train, y_train) = load_diabetes().unwrap();

    // create logistic regression model
    let mut log_model = Logistic::new(
        x_train.clone(),
        y_train.clone(),
        sigmoid_vec,
        0.01
    ).unwrap();

    log_model.sgd(2000, true, 5);

    let x_test = x_train.batch(5).unwrap();
    let y_test = y_train.batch(5).unwrap();
    let y_pred = log_model.predict(x_test[30].clone());
    println!("Actual: {:?}", y_test[30]);
    println!("Prediction: {:?}", y_pred.values());

    let loss = mse(&y_test[30], &y_pred).unwrap(); 
    println!("LOSS: {:?}", loss);

}


fn breast_cancer_model() {

    // load data
    let (x_train, y_train) = load_breast_cancer().unwrap();

    // create logistic regression model
    let mut log_model = Logistic::new(
        x_train.clone(),
        y_train.clone(),
        sigmoid_vec,
        0.001
    ).unwrap();

    log_model.sgd(1500, true, 5);

    let sample_index = 100;
    let x_test = x_train.batch(5).unwrap();
    let y_test = y_train.batch(5).unwrap();
    let y_pred = log_model.predict(x_test[sample_index].clone());
    println!("Actual: {:?}", y_test[sample_index]);
    println!("Prediction: {:?}", y_pred.values());

    let loss = mse(&y_test[sample_index], &y_pred).unwrap(); 
    println!("LOSS: {:?}", loss);
}


fn main() {

    //diabetes_model(); 
    //breast_cancer_model(); 
   
    // load data
    let (x_train, y_train) = load_iris().unwrap();

    // create logistic regression model
    let mut log_model = Logistic::new(
        x_train.clone(),
        y_train.clone(),
        softmax,
        0.001
    ).unwrap();

    log_model.sgd(1000, true, 3);


    let sample_index = 100;
    let x_test = x_train.batch(3).unwrap();
    let y_test = y_train.batch(3).unwrap();
    let y_pred = log_model.predict(x_test[sample_index].clone());
    println!("Actual: {:?}", y_test[sample_index]);
    println!("Prediction: {:?}", y_pred.values());

    let loss = mse(&y_test[sample_index], &y_pred).unwrap(); 
    println!("LOSS: {:?}", loss);  

}
