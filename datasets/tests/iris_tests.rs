
#[cfg(test)]
mod iris_tests {

    use datasets::iris::*;

    #[test]
    fn test_load_schema() {

        let iris_schema = load_iris_schema();
        let schema_fields = iris_schema.fields();
        let expected_fields = vec![
            "id",
            "sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm",
            "species_code",
            "species",
        ];

        let mut index = 0; 
        for field in schema_fields {
            assert_eq!(field.name(), expected_fields[index]);
            index += 1; 
        }

    }


    #[test]
    fn test_load_data() {
        let (x_train, y_train) = load_iris("data/iris.parquet").unwrap();
        println!("{:?}", x_train.shape().values()); 
        println!("{:?}", y_train.shape().values()); 
        assert_eq!(x_train.shape().values(), vec![150, 4]);
        assert_eq!(y_train.shape().values(), vec![150, 1]);
    }

}
