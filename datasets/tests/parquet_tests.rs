
#[cfg(test)]
mod parquet_tests {

    use datasets::utils::*; 
    use datasets::alzhiemers::*;
    use datasets::airfoil_noise::*;
    use std::{fs::File, path::Path};
    use parquet::errors::Result; 

    #[test]
    fn test_csv_to_parquet() -> Result<()> {

        convert_airfoil_csv_to_parquet();

        Ok(())

    }

}  
