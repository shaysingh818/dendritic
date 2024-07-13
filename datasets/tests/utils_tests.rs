
#[cfg(test)]
mod utils_tests {

    use datasets::utils::*; 
    use datasets::diabetes::*;
    use datasets::breast_cancer::*;
    use datasets::iris::*;
    use std::{fs::File, path::Path};
    use parquet::errors::Result; 
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    #[test]
    fn test_csv_to_parquet() -> Result<()> {

        let iris_schema = load_iris_schema();

        csv_to_parquet(
            iris_schema,
            "data/iris.csv",
            "data/iris.parquet"
        ); 

        let path = "data/diabetes.parquet";
        let file = File::open(path).unwrap();

        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_batch_size(5)
            .build()?;

        let batch = reader.next().unwrap().unwrap();
        let col1 = process_column(batch.clone(), "pregnancies");
        let col2 = process_column(batch.clone(), "glucose");
        let col3 = process_column(batch.clone(), "blood_pressure");
        let col4 = process_column(batch.clone(), "skin_thickness");
        let col5 = process_column(batch.clone(), "insulin");
        let col6 = process_column(batch.clone(), "bmi");
        let col8 = process_column(batch.clone(), "age");
        let col9 = process_column(batch.clone(), "outcome");

        let expected_col1 = vec![6.0, 1.0, 1.0, 0.0, 3.0];
        let expected_col2 = vec![148.0, 85.0, 89.0, 137.0, 78.0];
        let expected_col3 = vec![72.0, 66.0, 66.0, 40.0, 50.0]; 
        let expected_col4 = vec![35.0, 29.0, 23.0, 35.0, 32.0];
        let expected_col5 = vec![0.0, 0.0, 94.0, 168.0, 88.0];
        let expected_col6 = vec![33.6, 26.6, 28.1, 43.1, 31.0];
        let expected_col8 = vec![50.0, 31.0, 21.0, 33.0, 26.0];
        let expected_col9 = vec![1.0, 0.0, 0.0, 1.0, 1.0];

        assert_eq!(col1, expected_col1);
        assert_eq!(col2, expected_col2);
        assert_eq!(col3, expected_col3);
        assert_eq!(col4, expected_col4);
        assert_eq!(col5, expected_col5);
        assert_eq!(col6, expected_col6);
        assert_eq!(col8, expected_col8);
        assert_eq!(col9, expected_col9);


        Ok(())

    }



}
