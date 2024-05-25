use postgres::{Client, NoTls, Error};
use serde::{Serialize};
use arrow::json::*;
use std::sync::Arc;
use arrow_array::RecordBatch;
use arrow_schema::{Schema, Field, DataType};
use arrow_array::array::{Float64Array};


use crate::ndarray::ndarray::NDArray;
use crate::regression::logistic::Logistic;


#[derive(Serialize)]
pub struct Suicide {
    gun: i32,
    attack_type: i32,
    weapon_type: i32,
    suicide: i32
}

// #[derive(Debug, Clone)]
pub struct SuicideDetectionModel {
    pub records: Vec<Suicide>,
    view_name: String,
    database_name: String,
    pg_host: String,
    pg_user: String,
    pg_pass: String,
    model: Logistic
}


impl SuicideDetectionModel {

    pub fn new(
        pg_host: String, 
        pg_user: String, 
        pg_pass: String,
        database_name: String,
        view_name: String) -> Self {

        Self {
            records: Vec::new(),
            pg_host: pg_host,
            pg_user: pg_user,
            pg_pass: pg_pass,
            database_name: database_name,
            view_name: view_name,
            model: Logistic::new(
                 NDArray::new(vec![1, 1]).unwrap(),
                 NDArray::new(vec![1, 1]).unwrap(),
                 0.01
            ).unwrap()
        }
    }

    pub fn records(&self) -> &Vec<Suicide> {
        &self.records
    }

    pub fn database_name(&self) -> &str {
        &self.database_name
    }

    pub fn view_name(&self) -> &str {
        &self.view_name
    }

    pub fn load_records(&mut self) -> Result<(), Error> {

        let conn_str = format!(
            "host={} user={} dbname={} password={}",
            self.pg_host, self.pg_user, self.database_name, self.pg_pass 
        );

        let query_str = format!(
            "SELECT * FROM \"WAREHOUSE\".\"{}\"",
            self.view_name
        );

        let mut client = Client::connect(&conn_str, NoTls)?;

        let results = client.query(&query_str, &[])?;
        for row in results {

            let suicide = Suicide {
                gun: row.get(0),
                attack_type: row.get(1),
                weapon_type: row.get(2),
                suicide: row.get(4)
            };
            self.records.push(suicide);
        }

        Ok(())

    }

    pub fn create_batch(&mut self) -> RecordBatch {

        let schema = Schema::new(vec![
            Field::new("gun", DataType::Float64, false),
            Field::new("attack_type", DataType::Float64, false),
            Field::new("weapon_type", DataType::Float64, false),
            Field::new("suicide", DataType::Float64, false),
        ]);

        let mut decoder = ReaderBuilder::new(Arc::new(schema)).build_decoder().unwrap();
        decoder.serialize(&self.records).unwrap();
        decoder.flush().unwrap().unwrap()
    }

    pub fn process_features(&self, batch: RecordBatch) -> (NDArray<f64>, NDArray<f64>) {

        let mut guns: Vec<f64> = batch.column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        let mut attacks: Vec<f64> = batch.column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        let mut weapon_types: Vec<f64> = batch.column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect();


        let suicides: Vec<f64> = batch.column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect();


        let mut feature_vec: Vec<f64> = Vec::new();
        feature_vec.append(&mut guns);
        feature_vec.append(&mut attacks);
        feature_vec.append(&mut weapon_types);


        let input: NDArray<f64> = NDArray::array(vec![guns.len(), 3], feature_vec.clone()).unwrap();
        let output: NDArray<f64> = NDArray::array(vec![guns.len(), 1], suicides).unwrap(); 

        (input, output)
    }

    pub fn train_model(&mut self) {
        let batch = self.create_batch(); 
        let (inputs, outputs) = self.process_features(batch);
        self.model = Logistic::new(inputs, outputs, 0.1).unwrap();
        self.model.train(5000, true, 10);
    }

    pub fn save_parameters(&mut self, filepath: &str) -> std::io::Result<()> {
        self.model.save(filepath)?;
        Ok(())
    }

}

// #[cfg(test)]
// mod gtd_test {

//     use crate::models::gtd::*;

//     #[test]
//     fn test_load_records() {

//         let mut gtd_model = SuicideDetectionModel::new(
//             "192.168.4.37".to_string(),
//             "postgres".to_string(),
//             "dsm001$:".to_string(),
//             "GLOBAL_TERRORISM".to_string(),
//             "SUICIDE_DETECTION_FEATURES".to_string()
//         );

//         gtd_model.load_records().unwrap();
//         assert_eq!(gtd_model.records().len(), 722); 

//     }

//     #[test]
//     fn test_load_model() {

//         // let x: NDArray<f64> = NDArray::load("data/logistic_testing_data/inputs").unwrap();
//         // let y: NDArray<f64> = NDArray::load("data/logistic_testing_data/outputs").unwrap();

//         // let mut loaded_model = Logistic::load("gtd_iteration1", x, y, 0.01).unwrap();
//         // let results = loaded_model.forward().unwrap();

//         // println!("{:?}", loaded_model.outputs().values());
//         // println!("{:?}", results.values());
//     }
// }






