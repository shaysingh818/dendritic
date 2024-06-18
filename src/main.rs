pub mod ndarray;
pub mod regression;
pub mod loss;
pub mod models;
pub mod autodiff; 

use crate::models::exam_scores::*;

fn main()  {

    let mut model = ExamScoresModel::new(
        "<PG_HOST>".to_string(),
        "<PG_USER>".to_string(),
        "<PG_PASSWORD>".to_string(),
        "KAGGLE".to_string(),
        "MONERO".to_string(),
        0.001, 0.01
    );
    model.train();

}
