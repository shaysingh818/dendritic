use std::fs::File; 
use dendritic_preprocessing::standard_scalar::*;
use dendritic_ndarray::ndarray::NDArray;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::Result; 
use crate::utils::*;


/// loading the schema for the diabetes data
pub fn load_schema() -> Schema {
    Schema::new(vec![
        Field::new("pregnancies", DataType::Float64, false),
        Field::new("glucose", DataType::Float64, false),
        Field::new("blood_pressure", DataType::Float64, false),
        Field::new("skin_thickness", DataType::Float64, false),
        Field::new("insulin", DataType::Float64, false),
        Field::new("bmi", DataType::Float64, false),
        Field::new("pedigree_function", DataType::Float64, false),
        Field::new("age", DataType::Float64, false),
        Field::new("outcome", DataType::Float64, false)
    ])
}

/// unused but here for transparency on how datasets became parquet
pub fn convert_diabetes_csv_to_parquet() {

    let diabetes_schema = load_schema();

    csv_to_parquet(
        diabetes_schema,
        "data/diabetes.csv",
        "data/diabetes.parquet"
    ); 
}


/// Load x_train and y_train dataset for diabetes
pub fn load_diabetes(path: &str) -> Result<(NDArray<f64>, NDArray<f64>)> {

    let file = File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()?;

    let batch = reader.next().unwrap().unwrap();
    let (input, y_train) = select_features(
        batch.clone(),
        vec![
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "pedigree_function",
            //"age",
        ],
        "outcome"
    );

    let x_train = min_max_scalar(input).unwrap();
    Ok((x_train, y_train))

}




