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
    male: f64, 
    age: f64, 
    education: f64, 
    current_smoker: f64, 
    cigs_per_day: f64, 
    bp_meds: f64,
    prevalent_stroke: f64,
    prevalent_hyp: f64,
    diabetes: f64,
    to_chol: f64,
    sys_bp: f64,
    dia_bp: f64,
    bmi: f64,
    heart_rate: f64,
    glucose: f64,
    ten_year_chd: f64
}


pub struct HeartModel {
    pub records: Vec<Record>,
    pg_host: String,
    pg_user: String,
    pg_pass: String,
    database: String,
    view_name: String,
    learning_rate: f64,
    model: Logistic
}


impl HeartModel {

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
                &NDArray::new(vec![1, 1]).unwrap(),
                &NDArray::new(vec![1, 1]).unwrap(),
                sigmoid_vec,
                learning_rate
            ).unwrap()
        }
    }


    pub fn load_records(&mut self) -> Result<(), Error> {

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
            let patient_record = Record {
                male: row.get(0),
                age: row.get(1),
                education: row.get(2),
                current_smoker: row.get(3),
                cigs_per_day: row.get(4),
                bp_meds: row.get(5),
                prevalent_stroke: row.get(6),
                prevalent_hyp: row.get(7),
                diabetes: row.get(8),
                to_chol: row.get(9),
                sys_bp: row.get(10),
                dia_bp: row.get(11),
                bmi: row.get(12),
                heart_rate: row.get(13),
                glucose: row.get(14),
                ten_year_chd: row.get(15)
            };
            self.records.push(patient_record);
        }

        Ok(())
    }


    pub fn process_column(
        &self, 
        batch: 
        RecordBatch, 
        name: &str) -> Vec<f64> {

        batch.column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }


    pub fn create_batch(&mut self) -> RecordBatch {

        let schema = Schema::new(vec![
            Field::new("male", DataType::Float64, false),
            Field::new("age", DataType::Float64, false),
            Field::new("education", DataType::Float64, false),
            Field::new("current_smoker", DataType::Float64, false),
            Field::new("cigs_per_day", DataType::Float64, false),
            Field::new("bp_meds", DataType::Float64, false),
            Field::new("prevalent_stroke", DataType::Float64, false),
            Field::new("prevalent_hyp", DataType::Float64, false),
            Field::new("diabetes", DataType::Float64, false),
            Field::new("to_chol", DataType::Float64, false),
            Field::new("sys_bp", DataType::Float64, false),
            Field::new("dia_bp", DataType::Float64, false),
            Field::new("bmi", DataType::Float64, false),
            Field::new("heart_rate", DataType::Float64, false),
            Field::new("glucose", DataType::Float64, false),
            Field::new("ten_year_chd", DataType::Float64, false)
        ]);

        let mut decoder = ReaderBuilder::new(Arc::new(schema))
            .build_decoder()
            .unwrap();

        self.load_records().unwrap();
        decoder.serialize(&self.records).unwrap();
        decoder.flush().unwrap().unwrap()
    }


    pub fn select_features(
        &self,
        batch: RecordBatch,
        input_cols: Vec<&str>,
        output_col: &str
    ) -> (NDArray<f64>, NDArray<f64>) {

        let mut feature_vec: Vec<f64> = Vec::new();
        for col in &input_cols {
            let mut feature = self.process_column(batch.clone(), col);
            feature_vec.append(&mut feature);
        }

        let output_col = self.process_column(batch.clone(), output_col);

        let temp: NDArray<f64> = NDArray::array(
            vec![input_cols.len(), output_col.len()], 
            feature_vec.clone()
        ).unwrap();
        let input = temp.transpose().unwrap();
        
        let output: NDArray<f64> = NDArray::array(
            vec![output_col.len(), 1], 
            output_col
        ).unwrap();

        (input, output)
    }


    pub fn train(&mut self) {

        let batch = self.create_batch();
        let (x_train, y_train) = self.select_features(
            batch,
            vec![
                "male",
                "age",
                "current_smoker",
                "cigs_per_day",
                "bp_meds",
                "prevalent_stroke",
                "prevalent_hyp",
                "diabetes",
                "to_chol",
                "sys_bp",
                "dia_bp",
                "bmi",
                "heart_rate",
                "glucose"
            ],
            "ten_year_chd"
        );

        let x_train_processed = min_max_scalar(x_train).unwrap(); 
        self.model = Logistic::new(
            &x_train_processed, 
            &y_train, 
            sigmoid_vec,
            self.learning_rate
        ).unwrap();

        self.model.sgd(5000, true, 5); 
    }


    pub fn training_data(&mut self) -> (NDArray<f64>, NDArray<f64>) {
        let batch = self.create_batch();
        let (x_train, y_train) = self.select_features(
            batch,
            vec![
                "male",
                "age",
                "current_smoker",
                "cigs_per_day",
                "bp_meds",
                "prevalent_stroke",
                "prevalent_hyp",
                "diabetes",
                "to_chol",
                "sys_bp",
                "dia_bp",
                "bmi",
                "heart_rate",
                "glucose",
            ],
            "ten_year_chd"
        );

        let x_train_processed = min_max_scalar(x_train).unwrap();
        (x_train_processed, y_train)
    }

    pub fn save(&mut self, filepath: &str) -> std::io::Result<()> {
        self.model.save(filepath)?;
        Ok(())
    }


    pub fn load(&mut self, filepath: &str) {

        let batch = self.create_batch();
        let (x_train, y_train) = self.select_features(
            batch,
            vec![
                "male",
                "age",
                "current_smoker",
                "cigs_per_day",
                "bp_meds",
                "prevalent_stroke",
                "prevalent_hyp",
                "diabetes",
                "to_chol",
                "sys_bp",
                "dia_bp",
                "bmi",
                "heart_rate",
                "glucose",
            ],
            "ten_year_chd"
        );

        let x_train_processed = min_max_scalar(x_train).unwrap(); 

        self.model = Logistic::load(
            filepath,
            &x_train_processed, 
            &y_train,
            sigmoid_vec,
            self.learning_rate
        ).unwrap();
    }

    pub fn predict(&mut self, input: NDArray<f64>)  -> NDArray<f64> {
        self.model.predict(input) 
    }  

}
