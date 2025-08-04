
#[cfg(test)]
mod optimizer_tests {

    use std::fs;
    use std::fs::File;

    use ndarray::{arr2, Array2}; 

    use dendritic_optimizer::model::*; 
    use dendritic_optimizer::train::*;
    use dendritic_optimizer::optimizers::*;
    use dendritic_optimizer::optimizers::Optimizer;
    use dendritic_optimizer::regression::sgd::*;

    fn load_sample_data() -> (Array2<f64>, Array2<f64>) {

        let x = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ]);

        let y = arr2(&[[10.0], [12.0], [14.0], [16.0], [18.0]]);
        (x, y)
    }
 
    #[test]
    fn test_nesterov() -> std::io::Result<()> {


        let alpha = 0.001;
        let (x, y) = load_sample_data();
        let mut model = SGD::new(&x, &y, alpha).unwrap();
        let mut optimizer = Nesterov::default(&model);

        assert_eq!(optimizer.alpha, alpha);
        assert_eq!(optimizer.beta, 0.9);
        assert_eq!(optimizer.v.len(), model.graph.parameters().len());
        
        let expected_shapes = vec![(3, 1), (1, 1)];
        for (idx, item) in optimizer.v.iter().enumerate() {
            assert_eq!(item.dim(), expected_shapes[idx]);
        }

        for _ in 0..250 {
            model.graph.forward();
            model.graph.backward();
            optimizer.step(&mut model);
        }

        let loss_total = model.loss(); 
        assert_eq!(loss_total < 0.1, true);

        let predicted = model.predicted().mapv(|x| x.round());
        assert_eq!(predicted, y); 

        Ok(()) 
    }

    #[test]
    fn test_adagrad() -> std::io::Result<()> {

        let alpha = 0.5; // higher learning rate
        let (x, y) = load_sample_data();
        let mut model = SGD::new(&x, &y, alpha).unwrap();
        let mut optimizer = Adagrad::default(&model);

        assert_eq!(optimizer.alpha, alpha);
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.s.len(), model.graph.parameters().len());

        let expected_shapes = vec![(3, 1), (1, 1)];
        for (idx, item) in optimizer.s.iter().enumerate() {
            assert_eq!(item.dim(), expected_shapes[idx]);
        }

        for _ in 0..500 {
            model.graph.forward();
            model.graph.backward();
            optimizer.step(&mut model);
        }

        let loss_total = model.loss();
        assert_eq!(loss_total < 0.1, true);

        let predicted = model.predicted().mapv(|x| x.round());
        assert_eq!(predicted, y); 

        Ok(())

    }

    #[test]
    fn test_rmsprop() -> std::io::Result<()> {

        let alpha = 0.01; // higher learning rate
        let (x, y) = load_sample_data();
        let mut model = SGD::new(&x, &y, alpha).unwrap();
        let mut optimizer = RMSProp::default(&model);

        assert_eq!(optimizer.alpha, alpha);
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.decay_rate, 0.9);
        assert_eq!(optimizer.s.len(), model.graph.parameters().len());

        let expected_shapes = vec![(3, 1), (1, 1)];
        for (idx, item) in optimizer.s.iter().enumerate() {
            assert_eq!(item.dim(), expected_shapes[idx]);
        }

        for _ in 0..350 {
            model.graph.forward();
            model.graph.backward();
            optimizer.step(&mut model);
        }

        let loss_total = model.loss();
        assert_eq!(loss_total < 0.1, true);

        let predicted = model.predicted().mapv(|x| x.round());
        assert_eq!(predicted, y); 

        Ok(())
    }

    #[test]
    fn test_adadelta() -> std::io::Result<()> {

        let alpha = 0.9; // higher learning rate
        let (x, y) = load_sample_data();
        let mut model = SGD::new(&x, &y, alpha).unwrap();
        let mut optimizer = Adadelta::default(&model);

        assert_eq!(optimizer.y_s, 0.95);
        assert_eq!(optimizer.y_x, 0.95);
        assert_eq!(optimizer.epsilon, 1e-6);
        assert_eq!(optimizer.s.len(), model.graph.parameters().len());
        assert_eq!(optimizer.u.len(), model.graph.parameters().len());

        let expected_shapes = vec![(3, 1), (1, 1)];
        for (idx, item) in optimizer.s.iter().enumerate() {
            assert_eq!(item.dim(), expected_shapes[idx]);
        }

        for (idx, item) in optimizer.u.iter().enumerate() {
            assert_eq!(item.dim(), expected_shapes[idx]);
        }

        for _ in 0..350 {
            model.graph.forward();
            model.graph.backward();
            optimizer.step(&mut model);
        }

        let loss_total = model.loss();
        assert_eq!(loss_total < 204.0, true);

        Ok(())
    }

    #[test]
    fn test_adam() -> std::io::Result<()> {

        let alpha = 0.1; // higher learning rate
        let (x, y) = load_sample_data();
        let mut model = SGD::new(&x, &y, alpha).unwrap();
        let mut optimizer = Adam::default(&model);

        assert_eq!(optimizer.alpha, alpha);
        assert_eq!(optimizer.epsilon, 1e-6);
        assert_eq!(optimizer.y_s, 0.999);
        assert_eq!(optimizer.y_v, 0.9);
        assert_eq!(optimizer.k, 0);
        assert_eq!(
            optimizer.v_delta.len(), 
            model.graph.parameters().len()
        );
        assert_eq!(
            optimizer.s_delta.len(), 
            model.graph.parameters().len()
        );

        let expected_shapes = vec![(3, 1), (1, 1)];
        for (idx, item) in optimizer.v_delta.iter().enumerate() {
            assert_eq!(item.dim(), expected_shapes[idx]);
        }

        for (idx, item) in optimizer.s_delta.iter().enumerate() {
            assert_eq!(item.dim(), expected_shapes[idx]);
        }

        for _ in 0..350 {
            model.graph.forward();
            model.graph.backward();
            optimizer.step(&mut model);
        }

        let loss_total = model.loss();
        assert_eq!(loss_total < 0.1, true);

        let predicted = model.predicted().mapv(|x| x.round());
        assert_eq!(predicted, y);   
        assert_eq!(optimizer.k, 350); 

        Ok(())
    }

}

