use std::fs::File; 
use preprocessing::standard_scalar::*;
use ndarray::ndarray::NDArray;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::Result; 
use crate::utils::*;



/// loading the schema for the breast cancer data
pub fn load_alzheimers_schema() -> Schema {
    Schema::new(vec![
        Field::new("patient_id", DataType::Utf8, false),
        Field::new("age", DataType::Float64, false),
        Field::new("gender", DataType::Float64, false),
        Field::new("ethnicity", DataType::Float64, false),
        Field::new("education_level", DataType::Float64, false),
        Field::new("bmi", DataType::Float64, false),
        Field::new("smoking", DataType::Float64, false),
        Field::new("alchohol_consumption", DataType::Float64, false),
        Field::new("physical_activity", DataType::Float64, false),
        Field::new("diet_quality", DataType::Float64, false),
        Field::new("sleep_quality", DataType::Float64, false),
        Field::new("family_history", DataType::Float64, false),
        Field::new("cardiovascular_disease", DataType::Float64, false),
        Field::new("diabetes", DataType::Float64, false),
        Field::new("depression", DataType::Float64, false),
        Field::new("head_injury", DataType::Float64, false),
        Field::new("hyptertension", DataType::Float64, false),
        Field::new("systolic_bp", DataType::Float64, false),
        Field::new("distolic_dp", DataType::Float64, false),
        Field::new("cholesterol_total", DataType::Float64, false),
        Field::new("cholesterol_ldl", DataType::Float64, false),
        Field::new("cholesterol_hdl", DataType::Float64, false),
        Field::new("cholesterol_tryglicerides", DataType::Float64, false),
        Field::new("mmse", DataType::Float64, false),
        Field::new("functional_assesment", DataType::Float64, false),
        Field::new("memory_complaints", DataType::Float64, false),
        Field::new("behavorial_problems", DataType::Float64, false),
        Field::new("adl", DataType::Float64, false),
        Field::new("confusion", DataType::Float64, false),
        Field::new("disorientation", DataType::Float64, false), 
        Field::new("personality_changes", DataType::Float64, false),
        Field::new("difficulty_w_tasks", DataType::Float64, false),
        Field::new("forgetfullness", DataType::Float64, false),
        Field::new("diagnosis", DataType::Float64, false),
        Field::new("doctor", DataType::Utf8, false),
    ])
}


pub fn convert_alzhiemers_to_parquet() {

    let alz_schema = load_alzheimers_schema();

    csv_to_parquet(
        alz_schema,
        "data/alzheimers_disease_data.csv",
        "data/alzheimers.parquet"
    );
}


pub fn load_alzhiemers() -> Result<(NDArray<f64>, NDArray<f64>)> {
    
    /* switch to datasets/data directory */

    let path = "../../datasets/data/alzheimers.parquet";
    let file = File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()?;

    let batch = reader.next().unwrap().unwrap();
    let (input, y_train) = select_features(
        batch.clone(),
        vec![
            //"age",
            //"gender",
            //"ethnicity",
            //"education_level",
            "alchohol_consumption",
            //"bmi",
            "smoking",
            "physical_activity",
            "diet_quality",
            "sleep_quality",
            "family_history",
            //"cardiovascular_disease",
            //"diabetes",
            "depression",
            "head_injury",
            //"hyptertension",
            //"systolic_bp", 
            //"distolic_dp",
            //"cholesterol_total",
            //"cholesterol_ldl",
            //"cholesterol_hdl", 
            //"cholesterol_tryglicerides",
            "confusion",
            "disorientation",
            "personality_changes",
            "difficulty_w_tasks",
            "forgetfullness",
        ],
        "diagnosis"
    );

    let x_train = min_max_scalar(input).unwrap();
    Ok((x_train, y_train))

}




