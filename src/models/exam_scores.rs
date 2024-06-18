use postgres::{Client, NoTls, Error};
use serde::{Serialize};
use std::sync::Arc;
use std::fs::File; 
use arrow_schema::{Schema, Field, DataType};
use arrow_array::RecordBatch;
use arrow_array::array::{Float64Array};
use arrow::json::*;
use bigdecimal::BigDecimal;

use crate::ndarray::ndarray::NDArray;
use crate::regression::ridge::Ridge;


#[derive(Serialize, Debug)]
pub struct HousePrice {
    high: f64,
    low: f64,
    open: f64,
    market_cap: f64,
    volume: f64,
    close: f64
}


pub struct ExamScoresModel {
    pub records: Vec<HousePrice>,
    pg_host: String,
    pg_user: String,
    pg_pass: String,
    database: String,
    view_name: String
}


impl ExamScoresModel {

    pub fn new(
        pg_host: String, 
        pg_user: String, 
        pg_pass: String,
        database: String,
        view_name: String,
        lambda: f64, 
        learning_rate: f64) -> Self {

        Self {
            records: Vec::new(),
            pg_host: pg_host,
            pg_user: pg_user,
            pg_pass: pg_pass,
            database: database,
            view_name: view_name
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
            "SELECT * FROM \"PUBLIC\".\"{}\" LIMIT 10000",
            self.view_name
        );
        
        let mut client = Client::connect(&conn_str,  NoTls)?;
        let results = client.query(&query_str, &[])?;

        for row in results { 
            let house_price = HousePrice {
                high: row.get(0),
                low: row.get(1),
                open: row.get(2),
                market_cap: row.get(3),
                volume: row.get(4),
                close: row.get(5)
            };
            self.records.push(house_price);
        }

        Ok(())

    }

    pub fn create_batch(&mut self) -> RecordBatch {

        let schema = Schema::new(vec![
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("open", DataType::Float64, false),
            Field::new("market_cap", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
            Field::new("close", DataType::Float64, false)
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

        let mut high = self.process_column(batch.clone(), 0);
        let mut low = self.process_column(batch.clone(), 1);
        let mut open = self.process_column(batch.clone(), 2);
        //let mut market_cap = self.process_column(batch.clone(), 3);
        //let mut volume = self.process_column(batch.clone(), 4);
        let mut close = self.process_column(batch.clone(), 5);

        let mut feature_vec: Vec<f64> = Vec::new();
        feature_vec.append(&mut high);
        feature_vec.append(&mut low);
        feature_vec.append(&mut open);
        //feature_vec.append(&mut market_cap);
        //feature_vec.append(&mut volume);

        let input: NDArray<f64> = NDArray::array(
            vec![close.len(), 3], 
            feature_vec.clone()
        ).unwrap();




        let output: NDArray<f64> = NDArray::array(
            vec![close.len(), 1], 
            close
        ).unwrap(); 
        
        (input, output)
    }


    pub fn train(&mut self) {
        let batch = self.create_batch();
        let (x_train, y_train) = self.process_features(batch);
        let mut model = Ridge::new(x_train, y_train, 0.0001, 0.01).unwrap();
        model.train(10000, true);
    }


}
