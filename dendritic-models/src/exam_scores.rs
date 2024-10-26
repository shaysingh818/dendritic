use postgres::{Client, NoTls, Error};
use serde::{Serialize};
use std::sync::Arc;
use arrow_schema::{Schema, Field, DataType};
use arrow_array::RecordBatch;
use arrow_array::array::{Float64Array};
use arrow::json::*;

use dendritic_ndarray::ops::*;
use dendritic_ndarray::ndarray::NDArray;
use dendritic_regression::ridge::Ridge;
use dendritic_preprocessing::standard_scalar::*; 


#[derive(Serialize, Debug)]
pub struct HousePrice {
    median_age: f64,
    total_rooms: f64,
    total_bedrooms: f64,
    population: f64,
    households: f64,
    median_income: f64,
    house_value: f64
}


pub struct ExamScoresModel {
    pub records: Vec<HousePrice>,
    pg_host: String,
    pg_user: String,
    pg_pass: String,
    database: String,
    view_name: String,
    model: Ridge
}


impl ExamScoresModel {

    pub fn new(
        pg_host: String, 
        pg_user: String, 
        pg_pass: String,
        database: String,
        view_name: String) -> Self {

        Self {
            records: Vec::new(),
            pg_host: pg_host,
            pg_user: pg_user,
            pg_pass: pg_pass,
            database: database,
            view_name: view_name,
            model: Ridge::new(
                NDArray::new(vec![1, 1]).unwrap(),
                NDArray::new(vec![1, 1]).unwrap(),
                0.0001,
                0.01
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


    pub fn load_records_from_view(&mut self) -> Result<(), Error> {

        let conn_str = format!(
            "host={} user={} dbname={} password={}",
            self.pg_host, self.pg_user, self.database, self.pg_pass
        );

        let query_str = format!(
            "SELECT * FROM \"PUBLIC\".\"{}\" LIMIT 1000",
            self.view_name
        );
        
        let mut client = Client::connect(&conn_str,  NoTls)?;
        let results = client.query(&query_str, &[])?;

        for row in results { 
            let house_price = HousePrice {
                median_age: row.get(0),
                total_rooms: row.get(1),
                total_bedrooms: row.get(2),
                population: row.get(3),
                households: row.get(4),
                median_income: row.get(5),
                house_value: row.get(6)
            };
            self.records.push(house_price);
        }

        Ok(())

    }

    pub fn create_batch(&mut self) -> RecordBatch {

        let schema = Schema::new(vec![
            Field::new("median_age", DataType::Float64, false),
            Field::new("total_rooms", DataType::Float64, false),
            Field::new("total_bedrooms", DataType::Float64, false),
            Field::new("population", DataType::Float64, false),
            Field::new("households", DataType::Float64, false),
            Field::new("median_income", DataType::Float64, false),
            Field::new("house_value", DataType::Float64, false)
        ]);

        let mut decoder = ReaderBuilder::new(Arc::new(schema))
            .build_decoder()
            .unwrap();

        self.load_records_from_view().unwrap();
        decoder.serialize(&self.records).unwrap();
        decoder.flush().unwrap().unwrap()
    }


    pub fn process_features(
        &self, 
        batch: RecordBatch) -> (NDArray<f64>, NDArray<f64>) {

        let mut median_age = self.process_column(batch.clone(), 0);
        let mut total_rooms = self.process_column(batch.clone(), 1);
        let mut total_bedrooms = self.process_column(batch.clone(), 2);
        let mut population = self.process_column(batch.clone(), 3);
        let mut households = self.process_column(batch.clone(), 4);
        let mut med_income = self.process_column(batch.clone(), 5);
        let house_value = self.process_column(batch.clone(), 6);

        let mut feature_vec: Vec<f64> = Vec::new();
        feature_vec.append(&mut median_age);
        feature_vec.append(&mut total_rooms);
        feature_vec.append(&mut total_bedrooms);
        feature_vec.append(&mut population);
        feature_vec.append(&mut households);
        feature_vec.append(&mut med_income);

        let temp: NDArray<f64> = NDArray::array(
            vec![6, house_value.len()], 
            feature_vec.clone()
        ).unwrap();
        let input  = temp.transpose().unwrap();

        let output: NDArray<f64> = NDArray::array(
            vec![house_value.len(), 1], 
            house_value
        ).unwrap(); 
        
        (input, output)
    }

    pub fn training_data(&mut self) -> (NDArray<f64>, NDArray<f64>) {
        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        let x_train_processed = min_max_scalar(x_train).unwrap();
        //let y_train_processed = min_max_scalar(y_train).unwrap(); 
        (x_train_processed, y_train)
    }

    pub fn train(&mut self) {

        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        let x_train_processed = min_max_scalar(x_train).unwrap(); 
        //let y_train_processed = min_max_scalar(y_train).unwrap(); 

        self.model = Ridge::new(
            x_train_processed, 
            y_train, 
            0.00001, 
            0.3
        ).unwrap();

        self.model.sgd(1000, true, 5);
    }

    pub fn load(&mut self, filepath: &str) {

        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        let x_train_processed = min_max_scalar(x_train).unwrap(); 

        self.model = Ridge::load(
            filepath,
            x_train_processed, 
            y_train,
            0.0001, 0.1
        ).unwrap();
    }

    pub fn save(&mut self, filepath: &str) -> std::io::Result<()> {
        self.model.save(filepath)?;
        Ok(())
    }

    pub fn predict(&mut self, input: NDArray<f64>)  -> NDArray<f64> {
        self.model.predict(input) 
    }  


}
