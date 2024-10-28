use postgres::{Client, NoTls, Error};
use arrow_schema::{Schema, Field, DataType};
use arrow_array::RecordBatch;
use arrow_array::array::{Float64Array};
use arrow::json::*; 
use std::sync::Arc;
use serde::{Serialize};

use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;
use dendritic_regression::logistic::Logistic;
use dendritic_preprocessing::standard_scalar::*;
use dendritic_metrics::activations::*;

#[derive(Debug, serde::Deserialize, Serialize)]
pub struct Record {
    feature_0: f64,
    feature_1: f64, 
    feature_2: f64, 
    feature_3: f64, 
    feature_4: f64, 
    target: f64
}

pub struct ExampleModel {
    pub records: Vec<Record>,
    pg_host: String,
    pg_user: String,
    pg_pass: String,
    database: String,
    view_name: String,
    learning_rate: f64,
    model: Logistic
}


impl ExampleModel {

    pub fn new(
        pg_host: String, 
        pg_user: String, 
        pg_pass: String,
        database: String,
        view_name: String,
        learning_rate: f64) -> Self {

        Self {
            records: Vec::new(),
            pg_host: pg_host,
            pg_user: pg_user,
            pg_pass: pg_pass,
            database: database,
            view_name: view_name,
            learning_rate: learning_rate.clone(),
            model: Logistic::new(
                NDArray::new(vec![1, 1]).unwrap(),
                NDArray::new(vec![1, 1]).unwrap(),
                sigmoid_vec,
                learning_rate
            ).unwrap()
        }
    }


    pub fn process_column(
        &self, 
        batch: 
        RecordBatch, 
        index: usize) -> Vec<f64> {

        batch.column(index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }


    pub fn load_records(&mut self) -> Result<(), Error> {

        let conn_str = format!(
            "host={} user={} dbname={} password={}",
            self.pg_host, self.pg_user, self.database, self.pg_pass
        );

        let query_str = format!(
            "SELECT * FROM \"PUBLIC\".\"{}\"",
            self.view_name
        );
        
        let mut client = Client::connect(&conn_str,  NoTls)?;
        let results = client.query(&query_str, &[])?;

        for row in results { 
            let house_price = Record {
                feature_0: row.get(0),
                feature_1: row.get(1),
                feature_2: row.get(2),
                feature_3: row.get(3),
                feature_4: row.get(4),
                target: row.get(5)
            };
            self.records.push(house_price);
        }

        Ok(())

    }

    pub fn create_batch(&mut self) -> RecordBatch {

        let schema = Schema::new(vec![
            Field::new("feature_0", DataType::Float64, false),
            Field::new("feature_1", DataType::Float64, false),
            Field::new("feature_2", DataType::Float64, false),
            Field::new("feature_3", DataType::Float64, false),
            Field::new("feature_4", DataType::Float64, false),
            Field::new("target", DataType::Float64, false) 
        ]);

        let mut decoder = ReaderBuilder::new(Arc::new(schema))
            .build_decoder()
            .unwrap();

        self.load_records().unwrap();
        decoder.serialize(&self.records).unwrap();
        decoder.flush().unwrap().unwrap()
    }


    pub fn process_features(
        &self, 
        batch: RecordBatch) -> (NDArray<f64>, NDArray<f64>) {

        let mut feature_0 = self.process_column(batch.clone(), 0);
        let mut feature_1 = self.process_column(batch.clone(), 1);
        let mut feature_2 = self.process_column(batch.clone(), 2);
        let mut feature_3 = self.process_column(batch.clone(), 3);
        let mut feature_4 = self.process_column(batch.clone(), 4);
        let mut target = self.process_column(batch.clone(), 5);


        let mut feature_vec: Vec<f64> = Vec::new();
        feature_vec.append(&mut feature_0);
        feature_vec.append(&mut feature_1);
        feature_vec.append(&mut feature_2);
        feature_vec.append(&mut feature_3);
        feature_vec.append(&mut feature_4);


        let temp: NDArray<f64> = NDArray::array(
            vec![5, target.len()], 
            feature_vec.clone()
        ).unwrap();
        let input = temp.transpose().unwrap();
        

        let output: NDArray<f64> = NDArray::array(
            vec![target.len(), 1], 
            target
        ).unwrap(); 
        
        (input, output) 
    }


    pub fn training_data(&mut self) -> (NDArray<f64>, NDArray<f64>) {
        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        //let x_train_processed = min_max_scalar(x_train).unwrap();
        //let y_train_processed = min_max_scalar(y_train).unwrap(); 
        (x_train, y_train)
    }
    

    pub fn train(&mut self) {

        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        let x_train_processed = min_max_scalar(x_train).unwrap();


        //println!("{:?}", x_train.values()); 
        //println!("{:?}", x_train); 
        //let y_train_processed = min_max_scalar(y_train).unwrap(); 
        
        self.model = Logistic::new(
            x_train_processed, 
            y_train, 
            sigmoid_vec,
            self.learning_rate
        ).unwrap();

        self.model.sgd(1500, true, 5); 
    }


    pub fn save(&mut self, filepath: &str) -> std::io::Result<()> {
        self.model.save(filepath)?;
        Ok(())
    }

    pub fn load(&mut self, filepath: &str) {

        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        //let x_train_processed = standard_scalar(x_train).unwrap(); 

        self.model = Logistic::load(
            filepath,
            x_train, 
            y_train,
            sigmoid_vec,
            self.learning_rate
        ).unwrap();
    }

    pub fn predict(&mut self, input: NDArray<f64>)  -> NDArray<f64> {
        self.model.predict(input) 
    }  

}
