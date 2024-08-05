
#[cfg(test)]
mod parquet_tests {

    use datasets::utils::*; 
    use datasets::alzhiemers::*;
    use std::{fs::File, path::Path};
    use parquet::errors::Result; 

    #[test]
    fn test_csv_to_parquet() -> Result<()> {

        let schema = load_alzheimers_schema();

        csv_to_parquet(
            schema,
            "data/alzheimers_disease_data.csv",
	    "data/test.parquet"
        );


        Ok(())

    }

}  
