use std::fs::File; 
use ndarray::ndarray::NDArray;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::Result; 
use crate::utils::*;
use preprocessing::standard_scalar::*;

/// Load schema for housing data
pub fn load_housing_schema() -> Schema {
    Schema::new(vec![
        Field::new("CRIM", DataType::Float64, false),
        Field::new("ZN", DataType::Float64, false),
        Field::new("INDUS", DataType::Float64, false),
        Field::new("CHAS", DataType::Float64, false),
        Field::new("NX", DataType::Float64, false),
        Field::new("RM", DataType::Float64, false),
        Field::new("AGE", DataType::Float64, false),
        Field::new("DIS", DataType::Float64, false),
        Field::new("RAD", DataType::Float64, false),
        Field::new("TAX", DataType::Float64, false),
        Field::new("PTRATIO", DataType::Float64, false),
        Field::new("B", DataType::Float64, false),
        Field::new("LSTAT", DataType::Float64, false),
        Field::new("MEDV", DataType::Float64, false)
    ])
}

/// Utility for converting housing data to parquet file
pub fn convert_housing_csv_to_parquet() {

    let housing_schema = load_housing_schema();

    csv_to_parquet(
        housing_schema,
        "data/boston_housing.csv",
        "data/boston_housing.parquet"
    ); 
}

/// Load housing data from path
pub fn load_housing_data() -> Result<(NDArray<f64>, NDArray<f64>)> {
    
    /* switch to datasets/data directory */
    let path = "data/boston_housing.parquet";
    let file = File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()?;

    let batch = reader.next().unwrap().unwrap();
    let (input, y_train) = select_features(
        batch.clone(),
        vec![
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV"
        ],
        "MEDV"
    );

    let x_train = min_max_scalar(input).unwrap();
    Ok((x_train, y_train))

}
