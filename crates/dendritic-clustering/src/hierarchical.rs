use dendritic_ndarray::ndarray::NDArray;
use dendritic_ndarray::ops::*;


#[derive(Debug)]
pub struct HierarchicalClustering {
    pub data: NDArray<f64>,
    pub distance_matrix: NDArray<f64>,
    pub clusters: Vec<usize>,
    distance_metric: fn(
        y1: &NDArray<f64>, 
        y2: &NDArray<f64>
    ) -> Result<f64, String>
}


impl HierarchicalClustering {

    pub fn new(
        data: &NDArray<f64>, 
        distance_metric: fn(
            y1: &NDArray<f64>, 
            y2: &NDArray<f64>) -> Result<f64, String>
        ) -> Result<HierarchicalClustering, String> {

        let rows = data.shape().dim(0);
        if rows <= 1 {
            let msg = "Not enough rows in sample data";
            return Err(msg.to_string());
        }

        Ok(Self {
            data: data.clone(),
            distance_matrix: NDArray::new(vec![rows, rows]).unwrap(),
            clusters: Vec::new(),
            distance_metric: distance_metric
        })

    }

    pub fn distance_matrix(&self) -> &NDArray<f64> {
        &self.distance_matrix
    }

    pub fn clusters(&self) -> &Vec<usize> {
        &self.clusters
    }

    pub fn calculate_distance_matrix(&mut self) {
        let mut start_row = 1;
        let mut idx = 0;
        let mut temp_idx = 1;
        let mut start = self.data.shape().dim(0); 
        for row in 0..self.data.shape().dim(0) { 
            let x = self.data.axis(0, row).unwrap();
            for col in 0..start-1 {
                let y = self.data.axis(0, start_row).unwrap();
                let dist = (self.distance_metric)(&x, &y).unwrap();
                self.distance_matrix.set(vec![start_row, idx], dist).unwrap(); 
                start_row += 1; 
            }
            idx += 1; 
            temp_idx += 1; 
            start_row = temp_idx; 
            start -= 1;
        }
    }


    pub fn get_comparison_coords(
        &self,
        row: usize,
        min_coord: usize,
        max_coord: usize,
        d_matrix: &NDArray<f64>,
        coordinate: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {

        let x = d_matrix.axis(0, row).unwrap();
        let y = d_matrix.axis(1, row).unwrap();

        let p1_x = x.values()[coordinate[0]];
        let p2_x = x.values()[coordinate[1]];

        let p1_y = y.values()[coordinate[0]];
        let p2_y = y.values()[coordinate[1]];

        let mut c1: Vec<usize> = Vec::new();
        let mut c2: Vec<usize> = Vec::new();

        println!(
            "{:?} {:?} {:?} {:?}",
            p1_x, p2_x, p1_y, p2_y
        ); 

        if p1_y != 0.0 && p2_y != 0.0 {
            c1.push(min_coord);
            c1.push(row);
            c2.push(max_coord);
            c2.push(row);
        } else if p1_x != 0.0 && p2_x != 0.0 {
            c1.push(row);
            c1.push(min_coord);
            c2.push(row);
            c2.push(max_coord);
        } else if p2_x != 0.0 && p1_y != 0.0 {    
            c1.push(row);
            c1.push(min_coord);
            c2.push(max_coord);
            c2.push(row);
        } else if p1_x != 0.0 && p1_y != 0.0 {
            c1.push(min_coord);
            c1.push(row);
            c2.push(max_coord);
            c2.push(min_coord);
        }

        (c1, c2)
    }

    /*
    pub fn calc_dist_mat(
        &mut self,
        mut d_matrix: NDArray<f64>,
        coordinate: &Vec<usize>) -> NDArray<f64> {

        let min_coord = coordinate.into_iter().min().unwrap();
        let max_coord = coordinate.into_iter().max().unwrap();

        let drop_row = d_matrix.drop_axis(0, *max_coord).unwrap();
        let mut new_mat = drop_row.drop_axis(1, *max_coord).unwrap();

        println!("Executing... {:?}", coordinate);
         
        let rows = new_mat.shape().dim(0);
        for row in 0..rows {
            let item = d_matrix.axis(0, row).unwrap();
            let coord_check = !coordinate.iter().any(|&i| i == row);
            let clst_check = self.clusters.iter().any(|&i| i == row);

            if clst_check {

                let (dist, coord) = self.get_comparison_coords(
                    row,
                    *min_coord,
                    *max_coord,
                    &d_matrix,
                    &coordinate
                );

            } else if coord_check {

                let (dist, coord) = self.get_comparison_coords(
                    row,
                    *min_coord,
                    *max_coord,
                    &d_matrix,
                    &coordinate
                );

                new_mat.set(coord, dist).unwrap(); 
            }
        }

        self.clusters.push(*min_coord); 
        new_mat
    } */ 


    pub fn update_dist_mat(
        &mut self,
        mut d_matrix: NDArray<f64>,
        coordinate: &Vec<usize>) -> NDArray<f64> {

        let min_coord = coordinate.into_iter().min().unwrap();
        let max_coord = coordinate.into_iter().max().unwrap();

        let drop_row = d_matrix.drop_axis(0, *max_coord).unwrap();
        let mut new_mat = drop_row.drop_axis(1, *max_coord).unwrap();

        let mut rows = d_matrix.shape().dim(0);
        /*
        if self.clusters.len() > 0 {
            rows = new_mat.shape().dim(0); 
        } */

        for row in 0..rows {

            let row_check = !coordinate.iter().any(|&i| i == row);  
            let clst_check = self.clusters.iter().any(|&i| i == row);
            let mut coords: Vec<Vec<usize>> = Vec::new();

            /* generate all possible coordinates*/ 
            /* choose the pair that doesn't result in a distance of 0 */

            if row_check {

                let (c1, c2) = self.get_comparison_coords(
                    row,
                    *min_coord,
                    *max_coord,
                    &d_matrix,
                    &coordinate
                );

                let p1 = d_matrix.get(c1.clone());
                let p2 = d_matrix.get(c2.clone());
                let min = f64::min(*p1, *p2); 
                new_mat.set(c1, min);

                /*
                println!(
                    "{:?} -> ({:?} {:?})",
                    row, c1, c2
                ); */ 
                

 
            }
        
        }

        self.clusters.push(*min_coord);
        new_mat
    }

    pub fn find_min_coord(
        &self, 
        dist_matrix: &NDArray<f64>) -> Vec<usize> {
        let nonzero = dist_matrix.nonzero();
        let nzero_vals = nonzero.values();
        let min = nzero_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let indices = dist_matrix.value_indices(min);
        let idx = indices[0];
        let coordinate = dist_matrix.indices(idx).unwrap();
        coordinate
    }


    pub fn fit_transform(&mut self) {

        let mut d_matrix = self.distance_matrix.clone();
        let nonzero = d_matrix.nonzero();
        let nzero_vals = nonzero.values();
        let min = nzero_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let indices = d_matrix.value_indices(min);
        let idx = indices[0];
        let coordinate = d_matrix.indices(idx).unwrap();
        let cluster_idx = coordinate.iter().min().unwrap();


    }

}
