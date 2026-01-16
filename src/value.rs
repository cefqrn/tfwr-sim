use crate::context::{Context, EvaluationError};
use crate::statement::Block;

use std::cmp::Ordering;
use std::rc::Rc;

mod parsing;

#[derive(Clone, Debug)]
pub enum Value {
    None,
    String(String),
    Number(f64),
    Bool(bool),
    Function(Rc<Closure>),
    Tuple(Vec<Value>),
}

#[derive(Clone, Debug)]
pub struct Closure {
    pub parameters: Vec<String>,
    pub body: Block,
    pub base_context: Context,
    pub locals: Vec<String>,
}

impl From<Value> for bool {
    fn from(value: Value) -> Self {
        match value {
            Value::Bool(b) => b,
            Value::Number(n) => n != 0.,
            Value::Tuple(e) => !e.is_empty(),
            Value::None => false,
            _ => true,
        }
    }
}

impl From<&Value> for bool {
    fn from(value: &Value) -> Self {
        match value {
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.,
            Value::Tuple(e) => !e.is_empty(),
            Value::None => false,
            _ => true,
        }
    }
}

impl TryFrom<Value> for f64 {
    type Error = EvaluationError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Number(n) => Ok(n),
            Value::Bool(true) => Ok(1.),
            Value::Bool(false) => Ok(0.),
            _ => Err(EvaluationError),
        }
    }
}

impl TryFrom<&Value> for f64 {
    type Error = EvaluationError;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        match value {
            Value::Number(n) => Ok(*n),
            Value::Bool(true) => Ok(1.),
            Value::Bool(false) => Ok(0.),
            _ => Err(EvaluationError),
        }
    }
}

// no Eq since you can get NaN from inf - inf and inf through big number
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            (Self::Number(_) | Self::Bool(_), Self::Number(_) | Self::Bool(_)) => {
                f64::try_from(self).expect("checked") == f64::try_from(other).expect("checked")
            }
            (Self::Function(a), Self::Function(b)) => Rc::ptr_eq(a, b),
            (Self::Tuple(a), Self::Tuple(b)) => a == b,
            (Self::None, Self::None) => true,
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::String(l0), Self::String(r0)) => l0.partial_cmp(r0),
            (Self::Number(_) | Self::Bool(_), Self::Number(_) | Self::Bool(_)) => {
                f64::try_from(self)
                    .expect("checked")
                    .partial_cmp(&f64::try_from(other).expect("checked"))
            }
            (Self::Function(a), Self::Function(b)) => Rc::ptr_eq(a, b).then_some(Ordering::Equal),
            (Self::Tuple(a), Self::Tuple(b)) => a.partial_cmp(b),
            (Self::None, Self::None) => Some(Ordering::Equal),
            _ => None,
        }
    }
}
