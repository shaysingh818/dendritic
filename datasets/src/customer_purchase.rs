use std::fs::File; 
use ndarray::ndarray::NDArray;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::Result; 
use crate::utils::*;
use preprocessing::standard_scalar::*;

/// Load customer data schema
pub fn load_customer_schema() -> Schema {
    Schema::new(vec![
        Field::new("age", DataType::Float64, false),
        Field::new("gender", DataType::Float64, false),
        Field::new("annual_income", DataType::Float64, false),
        Field::new("number_of_purchases", DataType::Float64, false),
        Field::new("product_category", DataType::Float64, false),
        Field::new("time_spent_website", DataType::Float64, false),
        Field::new("loyalty_program", DataType::Float64, false),
        Field::new("discounts", DataType::Float64, false),
        Field::new("purchase_status", DataType::Float64, false)
    ])
}

/// Utility to convert customer data to parquet 
pub fn convert_customer_csv_to_parquet() {

    let iris_schema = load_customer_schema();

    csv_to_parquet(
        iris_schema,
        "data/customer_purchase_data.csv",
        "data/customer_purchase_data.parquet"
    ); 
}

/// Load customer data from path
pub fn load_customer_data() -> Result<(NDArray<f64>, NDArray<f64>)> {
    
    /* switch to datasets/data directory */
    let path = "data/customer_purchase_data.parquet";
    let file = File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()?;

    let batch = reader.next().unwrap().unwrap();
    let (input, y_train) = select_features(
        batch.clone(),
        vec![
            "age",
            "gender",
            "annual_income",
            "number_of_purchases",
            "product_category",
            "time_spent_website",
            "loyalty_program",
            "discounts",
        ],
        "purchase_status"
    );

    let x_train = min_max_scalar(input).unwrap();
    Ok((x_train, y_train))

}
