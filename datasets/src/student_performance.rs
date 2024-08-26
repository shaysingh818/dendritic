use std::fs::File; 
use ndarray::ndarray::NDArray;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::Result; 
use crate::utils::*;
use preprocessing::standard_scalar::*;


pub fn load_student_schema() -> Schema {
    Schema::new(vec![
        Field::new("student_id", DataType::Float64, false),
        Field::new("age", DataType::Float64, false),
        Field::new("gender", DataType::Float64, false),
        Field::new("ethnicity", DataType::Float64, false),
        Field::new("parental_education", DataType::Float64, false),
        Field::new("study_time_weekly", DataType::Float64, false),
        Field::new("absences", DataType::Float64, false),
        Field::new("tutoring", DataType::Float64, false),
        Field::new("parental_support", DataType::Float64, false),
        Field::new("extra_cirricular", DataType::Float64, false),
        Field::new("sports", DataType::Float64, false),
        Field::new("music", DataType::Float64, false),
        Field::new("volunteering", DataType::Float64, false),
        Field::new("gpa", DataType::Float64, false),
        Field::new("grade_class", DataType::Float64, false),
    ])
}


pub fn convert_student_csv_to_parquet() {

    let student_schema = load_student_schema();

    csv_to_parquet(
        student_schema,
        "data/student_performance.csv",
        "data/student_performance.parquet"
    ); 
}


pub fn load_student_data() -> Result<(NDArray<f64>, NDArray<f64>)> {
    
    /* switch to datasets/data directory */
    let path = "data/student_performance.parquet";
    let file = File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .build()?;

    let batch = reader.next().unwrap().unwrap();
    let (input, y_train) = select_features(
        batch.clone(),
        vec![
            "age",
            "gender",
            "ethnicity",
            "parental_education",
            "study_time_weekly", 
            "absences",
            "tutoring",
            "parental_support", 
            "extra_cirricular",
            "sports",
            "music",
            "volunteering",
            "gpa",
        ],
        "grade_class"
    );

    let x_train = min_max_scalar(input).unwrap();
    Ok((x_train, y_train))

}
