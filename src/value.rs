use crate::evaluation::{Context, EvaluationError};
use crate::parsing;
use crate::statement::Block;
use parsing::{ParseError, ParseInput, ParseResult, Parser, Predicate};

use std::cmp::Ordering;
use std::collections::HashSet;
use std::rc::Rc;

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
    pub locals: HashSet<String>,
}

impl From<Value> for bool {
    fn from(value: Value) -> Self {
        match value {
            Value::Bool(b) => b,
            Value::Number(n) => n != 0.,
            Value::Tuple(e) => !e.is_empty(),
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

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            (Self::Number(_) | Self::Bool(_), Self::Number(_) | Self::Bool(_)) => {
                f64::try_from(self).expect("checked") == f64::try_from(other).expect("checked")
            }
            (Self::Function(a), Self::Function(b)) => Rc::ptr_eq(a, b),
            (Self::Tuple(a), Self::Tuple(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

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
            _ => None,
        }
    }
}

pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, Value> {
    let none = "None".map(|_| Value::None);
    let true_ = "True".map(|_| Value::Bool(true));
    let false_ = "False".map(|_| Value::Bool(false));

    let keyword = true_
        .or(false_)
        .or(none)
        .followed_by(parsing::identifier_boundary);

    number.or(string).or(keyword).try_parse(input)
}

fn number(input: ParseInput<'_>) -> ParseResult<'_, Value> {
    let digit = Predicate(&|c| c.is_ascii_digit());

    let ((whole, fractional), input) = digit
        .any_amount()
        .and('.'.before(digit.at_least_one()).maybe())
        .try_parse(input)?;

    let mut result = whole;
    if let Some((d, ds)) = fractional {
        result.push('.');
        result.push(d);
        result.extend(ds);
    } else if result.is_empty() {
        Err(ParseError)?;
    }
    let result = String::from_iter(result).parse().expect("numbers and dot");

    Ok((Value::Number(result), input))
}

fn string(input: ParseInput<'_>) -> ParseResult<'_, Value> {
    let escaped = |input| {
        let (_, rest) = '\\'.and(Predicate(&|_| true)).try_parse(input)?;
        let taken = &input.s[..input.s.len() - rest.s.len()];

        Ok((taken, rest))
    };

    let unescaped = |input| {
        let (_, rest) = Predicate(&|c| c != '"').try_parse(input)?;
        let taken = &input.s[..input.s.len() - rest.s.len()];

        Ok((taken, rest))
    };

    let (s, rest) = '"'
        .before(escaped.or(unescaped).any_amount())
        .followed_by('"')
        .try_parse(input)?;
    Ok((Value::String(s.join("")), rest))
}
