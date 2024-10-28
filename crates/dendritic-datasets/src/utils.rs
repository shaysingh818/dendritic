use std::path::Path;
use std::fs::File; 
use std::sync::Arc;
use arrow_csv::ReaderBuilder; 
use arrow_array::RecordBatch;
use arrow_array::array::{Float64Array};
use arrow_schema::{Schema}; 
use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;

use parquet::{
    basic::Compression,
    arrow::ArrowWriter,
    file::{
        properties::WriterProperties,
    }
};


/// General utility for converting csv files to parquet
pub fn csv_to_parquet(schema: Schema, filepath: &str, outpath: &str) {

    let path = Path::new(outpath);
    let file = File::open(filepath).unwrap();
    let out_file = File::create(&path).unwrap();

    let mut reader = ReaderBuilder::new(Arc::new(schema.clone()))
        .with_header(true)
        .build(file)
        .unwrap();

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();


    //let batch = reader.next().unwrap().unwrap();
    let mut writer = ArrowWriter::try_new(
        out_file,
        schema.into(),
        Some(props)
    ).unwrap();

    while let Some(batch) = reader.next() {
        let batch = batch.unwrap();
        writer.write(&batch).expect("WRITING BATCH"); 
    }

    writer.close().unwrap(); 

}

/// Select specific features of dataset for training
pub fn select_features(
    batch: RecordBatch,
    input_cols: Vec<&str>,
    output_col: &str
) -> (NDArray<f64>, NDArray<f64>) {


    let mut feature_vec: Vec<f64> = Vec::new();
    let output_col = process_column(batch.clone(), output_col);

    for col in &input_cols {
        let mut feature = process_column(batch.clone(), col);
        feature_vec.append(&mut feature);
    }

    let temp: NDArray<f64> = NDArray::array(
        vec![input_cols.len(), output_col.len()], 
        feature_vec.clone()
    ).unwrap();
    let input = temp.transpose().unwrap();
    
    let output: NDArray<f64> = NDArray::array(
        vec![output_col.len(), 1], 
        output_col
    ).unwrap();

    (input, output)
}


/// Needs to take in any type conversion at some point
pub fn process_column(batch: RecordBatch, name: &str) -> Vec<f64> {

    batch.column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .flatten()
        .collect()
}



