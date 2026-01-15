use crate::value::Value;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
pub struct EvaluationError;

pub type Variable = Rc<RefCell<Option<Value>>>;
pub type Context = HashMap<String, Variable>;

pub fn add(context: &mut Context, name: String, var: Variable) {
    context.insert(name, var);
}

pub fn declare(context: &mut Context, name: String) {
    add(context, name, Rc::new(RefCell::new(None)));
}

pub fn assign(context: &mut Context, name: String, value: Value) {
    context
        .entry(name)
        .or_insert_with(|| Rc::new(RefCell::new(None)))
        .replace(Some(value));
}

#[must_use]
pub fn capture(context: &Context, name: &str) -> Variable {
    let top = context.get(name).expect("already added to context");
    top.clone()
}
