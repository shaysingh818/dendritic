
#[cfg(test)]
mod utf_ops {

    use ndarray::ndarray::NDArray;
    use ndarray::ops::*; 

    #[test]
    fn test_unique_str_vals() {

        let features: NDArray<&str> = NDArray::array(
            vec![14, 4],
            vec![
                "Sunny", "Hot", "High", "Weak",
                "Sunny", "Hot", "High", "Strong",
                "Overcast", "Hot", "High", "Weak",
                "Rain", "Mild", "High", "Weak",
                "Rain", "Cool", "Normal", "Weak",
                "Rain", "Cool", "Normal", "Strong",
                "Overcast", "Cool", "Normal", "Strong",
                "Sunny", "Mild", "High", "Weak",
                "Sunny", "Cool", "Normal", "Weak",
                "Rain", "Mild", "Normal", "Weak",
                "Sunny", "Mild", "Normal", "Strong",
                "Overcast", "Mild", "High", "Strong",
                "Overcast", "Hot", "Normal", "Weak",
                "Rain", "Mild", "High", "Strong"
            ],
        ).unwrap();

        let outputs: NDArray<&str> = NDArray::array(
            vec![14, 1],
            vec![
                "No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"
            ]
        ).unwrap();


        let outlook = features.axis(1, 0).unwrap();
        let temperature = features.axis(1, 1).unwrap();
        let humidity = features.axis(1, 2).unwrap();
        let wind = features.axis(1, 3).unwrap();

        let outlook_vals = outlook.unique();
        let temperature_vals = temperature.unique();
        let humidity_vals = humidity.unique();
        let wind_vals = wind.unique();
        let output_vals = outputs.unique();

        assert_eq!(output_vals, vec!["No", "Yes"]);
        assert_eq!(outlook_vals, vec!["Sunny", "Overcast", "Rain"]);
        assert_eq!(temperature_vals, vec!["Hot", "Mild", "Cool"]);
        assert_eq!(humidity_vals, vec!["High", "Normal"]);
        assert_eq!(wind_vals, vec!["Weak", "Strong"]);
    }

    #[test]
    fn test_count_op() {
        
        let outputs: NDArray<&str> = NDArray::array(
            vec![14, 1],
            vec![
                "No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"
            ]
        ).unwrap();

        let expected_counts: Vec<usize> = vec![5, 9];

        let mut index = 0; 
        let vals = outputs.counts();
        for count in &vals {
            assert_eq!(count, &expected_counts[index]); 
            index += 1; 
        }
    
    }



}

