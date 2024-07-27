use datasets::iris::*;
use datasets::breast_cancer::*;
use datasets::diabetes::*;
use regression::logistic::*;
use trees::decision_tree::*;
use preprocessing::encoding::*;
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


fn iris_model() {

    // load data
    let (x_train, y_train) = load_iris().unwrap();

    // encode the target variables
    let mut encoder = OneHotEncoding::new(y_train.clone()).unwrap();
    let y_train_encoded = encoder.transform();

    // create logistic regression model
    let mut log_model = MultiClassLogistic::new(
        x_train.clone(),
        y_train_encoded.clone(),
        softmax,
        0.1
    ).unwrap();

    log_model.sgd(500, true, 5);

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
 
    // load data

    let sample_index = 98;
    let (x_train_test, y_train_test) = load_iris().unwrap();
    let (x_train, y_train) = load_all_iris().unwrap();
    let mut model = DecisionTreeClassifier::new(3, 3);
    model.fit(x_train.clone(), y_train);

    let x_batch = x_train.batch(5).unwrap();
    let y_actual = y_train_test.batch(5).unwrap();

    let predictions = model.predict(x_batch[sample_index].clone());
    println!("{:?}", y_actual[sample_index]);
    println!("Predict: {:?}", predictions); 

    


}
