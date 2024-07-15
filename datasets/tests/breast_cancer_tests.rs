
#[cfg(test)]
mod breast_cancer_tests {

    use datasets::breast_cancer::*;
    use arrow_schema::{DataType};

    #[test]
    fn test_load_breast_cancer_schema() {

       let iris_schema = load_breast_cancer_schema();
       let schema_fields = iris_schema.fields();
       let expected_fields = vec![
            "id",
            "diagnosis",
            "radius_mean",
            "texture_mean",
            "perimiter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave_points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se", 
            "concavity_se",
            "concave_points_se",
            "symmetry_se",
            "fractal_dimensions_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave_points_worst", 
            "symmetry_worst",
            "fractal_dimension_worst",
            "diagnosis_code"
        ];


        let mut index = 0; 
        for field in schema_fields {

            if index == 0 || index == 1 {
                assert_eq!(field.data_type(), &DataType::Utf8);
            } else {
                assert_eq!(field.data_type(), &DataType::Float64);
            }

            assert_eq!(field.name(), expected_fields[index]);
            index += 1; 
        }

    }

    #[test]
    fn test_load_data() {
        let (x_train, y_train) = load_breast_cancer().unwrap();
        assert_eq!(x_train.shape().values(), vec![569, 12]);
        assert_eq!(y_train.shape().values(), vec![569, 1]);
    }

}
