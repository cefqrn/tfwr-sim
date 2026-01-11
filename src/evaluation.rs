use crate::statement::Statement;
use crate::value::Value;

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug)]
pub struct EvaluationError;

pub type Variable = Rc<RefCell<Option<Value>>>;
pub type Context = HashMap<String, (Variable, Vec<Variable>)>;

pub fn declare(context: &mut Context, name: String) {
    context.insert(name, (Rc::new(RefCell::new(None)), Vec::new()));
}

pub fn declare_locally(context: &mut Context, name: &str) {
    let (top, stack) = context.get_mut(name).expect("already added to context");
    stack.push(std::mem::replace(top, Rc::new(RefCell::new(None))));
}

pub fn assign(context: &mut Context, name: &str, value: Value) {
    let (top, _) = context.get_mut(name).expect("already added to context");
    top.replace(Some(value));
}

#[must_use]
pub fn assigned_to_in(statements: &[Statement]) -> Vec<&str> {
    // TODO: globals
    let mut result = Vec::new();
    for s in statements {
        match s {
            Statement::Assignment(name, _) => result.push(name.as_str()),
            Statement::Def(name, _, _) => result.push(name.as_str()),
            Statement::If(if_branches, else_branch) => {
                for (_, body) in if_branches {
                    result.extend(assigned_to_in(body));
                }

                result.extend(assigned_to_in(else_branch));
            }
        }
    }

    result
}
