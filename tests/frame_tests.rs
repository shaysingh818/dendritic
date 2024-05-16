// use steepgrad::ndarray;

#[cfg(test)]
mod frame_test {


    use std::fs::File;
    use std::sync::Arc;
    use arrow::csv::*;
    use arrow_schema::{Schema, Field, DataType};
    use arrow_array::types::ByteArrayType;
    use arrow_array::cast::AsArray;
    use arrow_array::array::{StringArray, Float64Array};

    #[test]
    fn test_create_frame() {

        let schema = Schema::new(vec![
            Field::new("index", DataType::Utf8, false),
            Field::new("movie_name", DataType::Utf8, false),
            Field::new("year_of_release", DataType::Utf8, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("run_time", DataType::Utf8, false),
            Field::new("genre", DataType::Utf8, false),
            Field::new("imdb_rating", DataType::Float64, true),
            Field::new("votes", DataType::Utf8, true),
            Field::new("gross_total", DataType::Utf8, true),
        ]);

        let file_path = "data/movies.csv";
        let file = File::open(file_path).unwrap();
        let mut csv = ReaderBuilder::new(Arc::new(schema))
            .with_header(true)
            .build(file)
            .unwrap();

        let batch = csv.next().unwrap().unwrap();

        let mut ratings: Vec<f64> = batch.column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect();


        let mut votings: Vec<f64> = batch.column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        let mut feature_vec: Vec<f64> = Vec::new();
        feature_vec.append(&mut votings);
        feature_vec.append(&mut ratings);

        assert_eq!(feature_vec.len(), 198);

    }

}