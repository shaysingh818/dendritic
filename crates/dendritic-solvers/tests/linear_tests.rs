
#[cfg(test)]
mod linear_solver_tests {
   
    use ndarray::prelude::*; 
    use ndarray::{arr2, Array}; 
    use dendritic_solvers::linear::*;  

    #[test]
    fn test_linear_solver_create() -> Result<(), Box<dyn std::error::Error>> {

        let X = arr2(&[[2.0, 1.0], [5.0, 7.0]]);
        let b = arr2(&[[11.0], [13.0]]);
        
        let solver = LinearSolver::new(&X, &b, 0.01)?; 
        assert_eq!(solver.X.shape(), &[2, 2]);
        assert_eq!(solver.b.shape(), &[2, 1]);
        assert_eq!(solver.parameters.shape(), &[2, 1]);

        let X_bad = arr2(&[[2.0, 1.0], [5.0, 7.0], [1.0, 1.0]]);
        let b_bad = arr2(&[[11.0, 11.0], [13.0, 12.0]]);
        let b_bad_two = arr2(&[[11.0], [13.0], [12.0]]);

        let solver_bad = LinearSolver::new(&X_bad, &b, 0.01);
        let solver_bad_2 = LinearSolver::new(&X, &b_bad, 0.01);
        let solver_bad_3 = LinearSolver::new(&X, &b_bad_two, 0.01);

        assert_eq!(
            solver_bad.unwrap_err(),
            "Input must be a square matrix 3 != 2"
        );

        assert_eq!(
            solver_bad_2.unwrap_err(),
            "Target vector can't have more than 1 column: 2"
        );

        assert_eq!(
            solver_bad_3.unwrap_err(),
            "Solution set rows not equal 2 != 3"
        );
 
        Ok(())
    }

    #[test]
    fn test_gauss_solver() -> Result<(), Box<dyn std::error::Error>> {

        let X = arr2(&[
            [10.0, -1.0, 2.0, 0.0], 
            [-1.0, 11.0, -1.0, 3.0],
            [2.0, -1.0, 10.0, -1.0],
            [0.0, 3.0, -1.0, 8.0]
        ]);

        let b = arr2(&[[6.0], [25.0], [-11.0], [15.0]]);
        
        let mut solver = LinearSolver::new(&X, &b, 0.00)?;
        solver.gauss_seidal()?;

        assert_eq!(
            solver.parameters, 
            arr2(&[[1.0], [2.0], [-1.0], [1.0]])
        );
 
        Ok(())
    }


    #[test]
    fn test_sor_solver() -> Result<(), Box<dyn std::error::Error>> {

        let X = arr2(&[
            [4.0, -1.0, -6.0, 0.0], 
            [-5.0, -4.0, 10.0, 8.0],
            [0.0, 9.0, 4.0, -2.0],
            [1.0, 0.0, -7.0, 5.0]
        ]);

        let b = arr2(&[[2.0], [21.0], [-12.0], [-6.0]]);
        
        let mut solver = LinearSolver::new(&X, &b, 1e-6)?;
        solver.sor(0.5)?;

        assert_eq!(
            solver.parameters, 
            arr2(&[
                [1.2490234375], 
                [-2.2448974609375], 
                [1.9687713623046879], 
                [0.9108547973632815]])
        ); 
 
        Ok(())
    }





}
