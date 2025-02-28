
/// Value node for computation graph
#[derive(Debug, Clone, Default)]
pub struct Value<T> {
    pub value: Rc<RefCell<T>>,
    pub forward_derivative: Rc<RefCell<T>>,
    pub central_derivative: Rc<RefCell<T>>,
    pub backward_derivative: Rc<RefCell<T>>,
    pub gradient: Rc<RefCell<T>>,
}


impl<T: Clone> Value<T> {

    /// Create new instance of value for comptuation graph
    pub fn new(value: &T) -> Value<T> {
        
        Value {
            value: Rc::new(RefCell::new(value)),
            gradient: Rc::new(RefCell::new(value)),
            forward_derivative: Rc::new(RefCell::new(value)),
            central_derivative: Rc::new(RefCell::new(value)),
            gradient: Rc::new(RefCell::new(value)),
        }
    }

    /// Get value associated with structure
    pub fn value(&self) -> &T {
        self.value.borrow()
    }

    /// Returns forward derivative of value
    pub fn forward(&self) -> &T {
        self.forward_derivative.borrow()
    }

    /// Returns central derivative of value
    pub fn central(&self) -> &T {
        self.central_derivative.borrow()
    }

    /// Returns backward derivative of value
    pub fn backward(&self) -> &T {
        self.backward_derivative.borrow()
    }

    /// Get gradient of value
    pub fn gradient(&self) -> T {
        self.gradient.borrow().clone()
    }

    /// Set value associated with structure
    pub fn set_value(&mut self, value: &T) {
        self.value.replace(value);
    }

    /// Set gradient of value in computation graph
    pub fn set_gradient(&mut self, value: &T) {
        self.gradient.replace(value);
    }

}
