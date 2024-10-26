
#[cfg(test)]
mod diabetes_tests {

    use dendritic_datasets::diabetes::*;
    use arrow_schema::{DataType};

    #[test]
    fn test_load_schema() {

        let diabetes_schema = load_schema();
        let schema_fields = diabetes_schema.fields();
        let expected_fields = vec![
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "pedigree_function",
            "age",
            "outcome"
        ];

        let mut index = 0; 
        for field in schema_fields {
            assert_eq!(field.data_type(), &DataType::Float64);
            assert_eq!(field.name(), expected_fields[index]);
            index += 1; 
        }

    }


    #[test]
    fn test_load_data() {
        let (x_train, y_train) = load_diabetes("data/diabetes.parquet").unwrap();
        assert_eq!(x_train.shape().values(), vec![532, 7]);
        assert_eq!(y_train.shape().values(), vec![532, 1]);
    }

}
