use postgres::{Client, NoTls, Error};
use arrow_schema::{Schema, Field, DataType};
use arrow_array::RecordBatch;
use arrow_array::array::{Float64Array};
use arrow::json::*; 
use std::sync::Arc;
use serde::{Serialize};
use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;
use dendritic_regression::linear::Linear;
use dendritic_preprocessing::standard_scalar::*; 


#[derive(Debug, serde::Deserialize, Serialize)]
pub struct Record {
    cylinders: f64, 
    displacement: f64, 
    horsepower: f64, 
    weight: f64, 
    acceleration: f64, 
    model_year: f64,
    origin: f64,
    mpg: f64
}

pub struct MPGModel {
    pub records: Vec<Record>,
    pg_host: String,
    pg_user: String,
    pg_pass: String,
    database: String,
    view_name: String,
    learning_rate: f64,
    model: Linear
}

impl MPGModel {

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
            model: Linear::new(
                &NDArray::new(vec![1, 1]).unwrap(),
                &NDArray::new(vec![1, 1]).unwrap(),
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
                cylinders: row.get(0),
                displacement: row.get(1),
                horsepower: row.get(2),
                weight: row.get(3),
                acceleration: row.get(4),
                model_year: row.get(5),
                origin: row.get(6),
                mpg: row.get(7)
            };
            self.records.push(house_price);
        }

        Ok(())

    }

    pub fn create_batch(&mut self) -> RecordBatch {

        let schema = Schema::new(vec![
            Field::new("cylinders", DataType::Float64, false),
            Field::new("displacement", DataType::Float64, false),
            Field::new("horsepower", DataType::Float64, false),
            Field::new("weight", DataType::Float64, false),
            Field::new("acceleration", DataType::Float64, false),
            Field::new("model_year", DataType::Float64, false),
            Field::new("origin", DataType::Float64, false),
            Field::new("mpg", DataType::Float64, false)
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

        let mut cylinders = self.process_column(batch.clone(), 0);
        let mut displacement = self.process_column(batch.clone(), 1);
        let mut horsepower = self.process_column(batch.clone(), 2);
        let mut weight = self.process_column(batch.clone(), 3);
        let mut acceleration = self.process_column(batch.clone(), 4);
        let mut model_year = self.process_column(batch.clone(), 5);
        let mut origin = self.process_column(batch.clone(), 6);
        let mut mpg = self.process_column(batch.clone(), 7);

        let mut feature_vec: Vec<f64> = Vec::new();
        feature_vec.append(&mut cylinders);
        feature_vec.append(&mut displacement);
        feature_vec.append(&mut horsepower);
        feature_vec.append(&mut weight);
        feature_vec.append(&mut acceleration);
        feature_vec.append(&mut model_year);
        feature_vec.append(&mut origin);

        let temp: NDArray<f64> = NDArray::array(
            vec![7, mpg.len()], 
            feature_vec.clone()
        ).unwrap();
        let input = temp.transpose().unwrap();
        

        let output: NDArray<f64> = NDArray::array(
            vec![mpg.len(), 1], 
            mpg
        ).unwrap();
        
        (input, output)
    }


    pub fn train(&mut self) {

        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        let x_train_processed = min_max_scalar(x_train).unwrap(); 
        let y_train_processed = min_max_scalar(y_train).unwrap(); 
        //println!("{:?}", y_train); 

        self.model = Linear::new(
            &x_train_processed, 
            &y_train_processed, 
            self.learning_rate
        ).unwrap();

        self.model.sgd(1000, true, 5); 
    }

    pub fn training_data(&mut self) -> (NDArray<f64>, NDArray<f64>) {
        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        let x_train_processed = min_max_scalar(x_train).unwrap();
        let y_train_processed = min_max_scalar(y_train).unwrap(); 
        (x_train_processed, y_train_processed)
    }


    pub fn save(&mut self, filepath: &str) -> std::io::Result<()> {
        self.model.save(filepath)?;
        Ok(())
    }

    pub fn load(&mut self, filepath: &str) {

        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        let x_train_processed = min_max_scalar(x_train).unwrap(); 
        let y_train_processed = min_max_scalar(y_train).unwrap(); 

        self.model = Linear::load(
            filepath,
            &x_train_processed, 
            &y_train_processed,
            self.learning_rate
        ).unwrap();
    }

    pub fn predict(&mut self, input: NDArray<f64>)  -> NDArray<f64> {
        self.model.predict(input) 
    }  


}
