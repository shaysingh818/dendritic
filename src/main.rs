use std::fs::File;
use std::sync::Arc;
use std::{error::Error, io, process};
use arrow_schema::{Schema, Field, DataType};
use arrow::csv::*;


fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello, world!");
    let file_path = "../../Infrastructure/data_lake/KAGGLE/higher_ed_employee_salaries.csv";

    let schema = Schema::new(vec![
        Field::new("city", DataType::Utf8, false),
        Field::new("lat", DataType::Float64, false),
        Field::new("lng", DataType::Float64, false),
    ]);

    let file = File::open("test/data/uk_cities.csv").unwrap();

    let mut csv = ReaderBuilder::new(Arc::new(schema)).build(file).unwrap();
    let batch = csv.next().unwrap().unwrap();


    Ok(())    
}
