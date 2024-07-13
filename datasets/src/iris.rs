use std::path::Path;
use std::fs::File; 
use std::sync::Arc;
use preprocessing::standard_scalar::*;
use ndarray::ndarray::NDArray;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::Result; 
use crate::utils::*;

use parquet::{
    basic::Compression,
    arrow::ArrowWriter,
    file::{
        properties::WriterProperties,
        writer::SerializedFileWriter
    }
};

pub fn load_iris_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("sepal_length_cm", DataType::Float64, false),
        Field::new("sepal_width_cm", DataType::Float64, false),
        Field::new("petal_length_cm", DataType::Float64, false),
        Field::new("petal_width_cm", DataType::Float64, false),
        Field::new("species_code", DataType::Float64, false),
        Field::new("species", DataType::Utf8, false)
    ])
}


/// unused but here for transparency on how datasets became parquet
pub fn convert_iris_csv_to_parquet() {

    let iris_schema = load_iris_schema();

    csv_to_parquet(
        iris_schema,
        "data/iris.csv",
        "data/iris.parquet"
    ); 
}


pub fn load_iris() -> Result<(NDArray<f64>, NDArray<f64>)> {
    
    /* switch to datasets/data directory */

    let path = "../../datasets/data/iris.parquet";
    let file = File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()?;

    let batch = reader.next().unwrap().unwrap();
    let (input, y_train) = select_features(
        batch.clone(),
        vec![
            "sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm",
        ],
        "species_code"
    );

    //let x_train = min_max_scalar(input).unwrap();
    Ok((input, y_train))

}


