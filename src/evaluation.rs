use crate::expression::{Expression, Operation};
use crate::statement::Statement;
use crate::value::Value;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
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

#[must_use]
pub fn assigned_to_in(statements: &[Statement]) -> HashSet<&str> {
    let mut result = HashSet::new();
    for s in statements {
        match s {
            Statement::Assignment(name, _) => {
                result.insert(name.as_str());
            }
            Statement::Def(name, _, _, _, captured) => {
                result.insert(name.as_str());
                result.extend(captured.iter().map(String::as_str));
            }
            Statement::If(if_branches, else_branch) => {
                for (_, body) in if_branches {
                    result.extend(assigned_to_in(body));
                }

                result.extend(assigned_to_in(else_branch));
            }
            Statement::Global(_) => {}
        }
    }

    result
}

fn identifiers_in(expression: &Expression) -> HashSet<&str> {
    let mut result = HashSet::new();
    match expression {
        Expression::Identifier(name) => {
            result.insert(name.as_str());
        }
        Expression::Literal(_) => {}
        Expression::Operation(Operation::Unary(_, x)) => {
            result.extend(identifiers_in(x));
        }
        Expression::Operation(Operation::Binary(_, x, y)) => {
            result.extend(identifiers_in(x));
            result.extend(identifiers_in(y));
        }
    }

    result
}

#[must_use]
pub fn referred_to_in(statements: &[Statement]) -> HashSet<&str> {
    let mut result = assigned_to_in(statements);
    for s in statements {
        match s {
            Statement::Assignment(_, value) => {
                identifiers_in(value);
            }
            Statement::Def(_, _, body, local, _) => {
                result.extend(
                    referred_to_in(body).difference(&local.iter().map(String::as_str).collect()),
                );
            }
            Statement::If(if_branches, else_branch) => {
                for (condition, body) in if_branches {
                    result.extend(identifiers_in(condition));
                    result.extend(referred_to_in(body));
                }

                result.extend(referred_to_in(else_branch));
            }
            Statement::Global(_) => {}
        }
    }

    result
}
