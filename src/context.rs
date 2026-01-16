use crate::value::Value;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
pub struct EvaluationError;

pub type Variable = Rc<RefCell<Option<Value>>>;

#[derive(Clone, Debug)]
pub struct Context {
    data: HashMap<String, Variable>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn declare(&mut self, identifier: String) {
        self.data.insert(identifier, Rc::new(RefCell::new(None)));
    }

    #[must_use]
    pub fn get(&self, identifier: &str) -> Option<Value> {
        self.data.get(identifier).and_then(|x| x.borrow().clone())
    }

    pub fn set(&mut self, identifier: String, value: Value) {
        self.data
            .entry(identifier)
            .or_insert_with(|| Rc::new(RefCell::new(None)))
            .replace(Some(value));
    }

    #[must_use]
    pub fn capture(&self, identifiers: &[String]) -> Option<Self> {
        identifiers
            .iter()
            .map(|identifier| {
                self.data
                    .get(identifier)
                    .map(|variable| (identifier.clone(), variable.clone()))
            })
            .collect::<Option<_>>()
            .map(|data| Self { data })
    }
}
