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


/// loading the schema for the breast cancer data
pub fn load_breast_cancer_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("diagnosis", DataType::Utf8, false),
        Field::new("radius_mean", DataType::Float64, false),
        Field::new("texture_mean", DataType::Float64, false),
        Field::new("perimiter_mean", DataType::Float64, false),
        Field::new("area_mean", DataType::Float64, false),
        Field::new("smoothness_mean", DataType::Float64, false),
        Field::new("compactness_mean", DataType::Float64, false),
        Field::new("concavity_mean", DataType::Float64, false),
        Field::new("concave_points_mean", DataType::Float64, false),
        Field::new("symmetry_mean", DataType::Float64, false),
        Field::new("fractal_dimension_mean", DataType::Float64, false),
        Field::new("radius_se", DataType::Float64, false),
        Field::new("texture_se", DataType::Float64, false),
        Field::new("perimeter_se", DataType::Float64, false),
        Field::new("area_se", DataType::Float64, false),
        Field::new("smoothness_se", DataType::Float64, false),
        Field::new("compactness_se", DataType::Float64, false),
        Field::new("concavity_se", DataType::Float64, false),
        Field::new("concave_points_se", DataType::Float64, false),
        Field::new("symmetry_se", DataType::Float64, false),
        Field::new("fractal_dimensions_se", DataType::Float64, false),
        Field::new("radius_worst", DataType::Float64, false),
        Field::new("texture_worst", DataType::Float64, false),
        Field::new("perimeter_worst", DataType::Float64, false),
        Field::new("area_worst", DataType::Float64, false),
        Field::new("smoothness_worst", DataType::Float64, false),
        Field::new("compactness_worst", DataType::Float64, false),
        Field::new("concavity_worst", DataType::Float64, false),
        Field::new("concave_points_worst", DataType::Float64, false), 
        Field::new("symmetry_worst", DataType::Float64, false),
        Field::new("fractal_dimension_worst", DataType::Float64, false),
        Field::new("diagnosis_code", DataType::Float64, false)
    ])
}

pub fn convert_breast_cancer_csv_to_parquet() {

    let breast_cancer_schema = load_breast_cancer_schema();

    csv_to_parquet(
        breast_cancer_schema,
        "data/breast_cancer.csv",
        "data/breast_cancer.parquet"
    );
}


pub fn load_breast_cancer() -> Result<(NDArray<f64>, NDArray<f64>)> {
    
    /* switch to datasets/data directory */

    let path = "data/breast_cancer.parquet";
    let file = File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()?;

    let batch = reader.next().unwrap().unwrap();
    let (input, y_train) = select_features(
        batch.clone(),
        vec![
            "radius_mean",
            "texture_mean",
            "smoothness_mean",
            "compactness_mean",
            "symmetry_mean", 
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "smoothness_se",
            "compactness_se",
            "symmetry_se",
            "fractal_dimensions_se"
        ],
        "diagnosis_code"
    );

    let x_train = min_max_scalar(input).unwrap();
    Ok((x_train, y_train))

}
