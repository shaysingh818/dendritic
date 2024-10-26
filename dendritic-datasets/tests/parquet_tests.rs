
#[cfg(test)]
mod parquet_tests {

    use dendritic_datasets::utils::*; 
    use dendritic_datasets::alzhiemers::*;
    use dendritic_datasets::airfoil_noise::*;
    use std::{fs::File, path::Path};
    use parquet::errors::Result; 

    #[test]
    fn test_csv_to_parquet() -> Result<()> {

        //convert_airfoil_csv_to_parquet();

        Ok(())

    }

}  
