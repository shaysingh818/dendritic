use steepgrad::dataframe;

#[cfg(test)]
mod frame_test {

    use crate::dataframe::frame::Frame;

    #[test]
    fn test_create_frame() {

        let frame = Frame::new().unwrap();
        println!("{:?}", frame.header_count())

    }

}