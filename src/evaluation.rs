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

pub fn assign(context: &mut Context, name: &str, value: Value) {
    let top = context.get_mut(name).expect("already added to context");
    top.replace(Some(value));
}

#[must_use]
pub fn capture(context: &Context, name: &str) -> Variable {
    let top = context.get(name).expect("already added to context");
    top.clone()
}
