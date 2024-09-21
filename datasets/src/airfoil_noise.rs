use std::env;
use std::path::{Path, PathBuf};
use std::fs::File; 
use ndarray::ndarray::NDArray;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::Result; 
use crate::utils::*;
use preprocessing::standard_scalar::*;


pub fn load_airfoil_schema() -> Schema {
    Schema::new(vec![
        Field::new("x0", DataType::Float64, false),
        Field::new("x1", DataType::Float64, false),
        Field::new("x2", DataType::Float64, false),
        Field::new("x3", DataType::Float64, false),
        Field::new("x4", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ])
}


pub fn convert_airfoil_csv_to_parquet() {

    let airfoil_schema = load_airfoil_schema();

    csv_to_parquet(
        airfoil_schema,
        "data/airfoil_noise_data.csv",
        "data/airfoil_noise_data.parquet"
    ); 
}


pub fn load_airfoil_data(path: &str) -> Result<(NDArray<f64>, NDArray<f64>)> {
    
    let file = File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()?;

    let batch = reader.next().unwrap().unwrap();
    let (input, y_train) = select_features(
        batch.clone(),
        vec![
            "x0",
            "x1",
            "x2",
            "x3",
            "x4",
            "y"
        ],
        "y"
    );


    //let x_train = min_max_scalar(input).unwrap();
    Ok((input, y_train))

}
