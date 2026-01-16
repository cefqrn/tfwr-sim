use crate::context;
use crate::expression;
use crate::value::{Closure, Value};
use crate::variable::VariableState;
use context::{Context, EvaluationError};
use expression::{Call, Expression};

use std::collections::HashMap;
use std::rc::Rc;

mod evaluation;
mod parsing;

#[derive(Clone, Debug)]
pub enum Statement {
    Assignment(AssignmentTarget, Expression),
    Global(String),
    Call(Call),
    Def(Definition),
    If(Vec<(Expression, Block)>, Block),
    While(Expression, Block),
    For(AssignmentTarget, Expression, Block),
}

#[derive(Clone, Debug)]
pub struct Definition {
    pub name: String,
    parameters: Vec<String>,
    body: Block,
    locals: Vec<String>,
    pub captured: Vec<String>,
}

#[derive(Clone, Debug)]
pub enum AssignmentTarget {
    Single(String),
    Multiple(Vec<AssignmentTarget>),
}

#[derive(Clone, Debug)]
pub struct Block(pub Vec<Statement>, pub Option<EndStatement>);

#[derive(Clone, Debug)]
pub enum EndStatement {
    Return(Expression),
    Continue,
    Break,
}

pub enum EndReason {
    Return(Value),
    Continue,
    Break,
    End,
}

impl Block {
    #[must_use]
    const fn new() -> Self {
        Self(Vec::new(), None)
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}
