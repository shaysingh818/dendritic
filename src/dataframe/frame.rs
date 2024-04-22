use std::collections::HashMap;


#[derive(Debug, Clone)]
pub struct Frame {
    headers: Vec<String>,
    header_count: usize,
    row_count: usize,
}


impl Frame {

    pub fn new() -> Result<Frame, String> {
        Ok(Self {
            headers: vec![],
            header_count: 0,
            row_count: 0,
        })
    }

    pub fn header_count(&self) -> usize {
        self.header_count
    }


}


