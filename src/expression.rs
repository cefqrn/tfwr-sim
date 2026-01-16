use crate::value::Value;

mod evaluation;
mod parsing;

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    Literal(Value),
    Identifier(String),
    Operation(Operation),
    Tuple(Vec<Expression>),
    Call(Call),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Call(pub Box<Expression>, pub Vec<Expression>);

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    Unary(UnaryOperation, Box<Expression>),
    Binary(BinaryOperation, Box<Expression>, Box<Expression>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOperation {
    Pos,
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOperation {
    Arithmetic(ArithmeticOperation),
    Logical(LogicalOperation),
    Comparison(ComparisonOperation),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithmeticOperation {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    FloorDiv,
    Exp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalOperation {
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonOperation {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    Ne,
}
