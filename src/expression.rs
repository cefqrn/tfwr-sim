use crate::evaluation::{Context, EvaluationError};
use crate::parsing;
use crate::value;
use parsing::{ParseInput, ParseResult, Parser};
use value::Value;

#[derive(Clone, Debug)]
pub enum Expression {
    Literal(Value),
    Identifier(String),
    Operation(Operation),
}

#[derive(Clone, Debug)]
pub enum Operation {
    Unary(UnaryOperation, Box<Expression>),
    Binary(BinaryOperation, Box<Expression>, Box<Expression>),
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOperation {
    Pos,
    Neg,
    Call,
}

#[derive(Clone, Copy, Debug)]
pub enum BinaryOperation {
    Arithmetic(ArithmeticOperation),
}

#[derive(Clone, Copy, Debug)]
pub enum ArithmeticOperation {
    Add,
    Sub,
    Mul,
    Div,
}

impl Expression {
    pub fn evaluate(self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Literal(v) => Ok(v),
            Self::Operation(op) => op.evaluate(context),
            Self::Identifier(n) => context.get(&n).map_or_else(
                || Err(EvaluationError),
                |v| v.borrow().clone().ok_or(EvaluationError),
            ),
        }
    }
}

impl Operation {
    pub fn evaluate(self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Unary(op, x) => {
                let x = x.evaluate(context)?;
                match op {
                    UnaryOperation::Pos => match x {
                        Value::Number(_) | Value::Bool(_) => Ok(x),
                        _ => Err(EvaluationError),
                    },
                    UnaryOperation::Neg => match x.as_num()? {
                        Value::Number(x) => Ok(Value::Number(-x)),
                        _ => Err(EvaluationError),
                    },
                    UnaryOperation::Call => Err(EvaluationError), // TODO
                }
            }
            Self::Binary(op, x, y) => {
                let x = x.evaluate(context)?;

                match op {
                    BinaryOperation::Arithmetic(op) => {
                        let Value::Number(x) = x.as_num()? else {
                            panic!("as_num should return a number")
                        };
                        let Value::Number(y) = y.evaluate(context)?.as_num()? else {
                            panic!("as_num should return a number")
                        };

                        match op {
                            ArithmeticOperation::Add => Ok(Value::Number(x + y)),
                            ArithmeticOperation::Sub => Ok(Value::Number(x - y)),
                            ArithmeticOperation::Mul => Ok(Value::Number(x * y)),
                            ArithmeticOperation::Div if y == 0. => Err(EvaluationError),
                            ArithmeticOperation::Div => Ok(Value::Number(x / y)),
                        }
                    }
                }
            }
        }
    }
}

pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let pos_neg = |input| {
        let pos = '+'.map(|_| UnaryOperation::Pos);
        let neg = '-'.map(|_| UnaryOperation::Neg);

        let (ops, input) = neg
            .or(pos)
            .followed_by(parsing::whitespace)
            .any_amount()
            .try_parse(input)?;
        let (result, input) = call.try_parse(input)?;

        let result = ops.into_iter().rev().fold(result, |acc, op| {
            Expression::Operation(Operation::Unary(op, Box::new(acc)))
        });

        Ok((result, input))
    };

    let mul_div = |input| {
        let mul = '*'.map(|_| ArithmeticOperation::Mul);
        let div = '/'.map(|_| ArithmeticOperation::Div);

        let ((initial_term, terms), input) = pos_neg
            .and(
                parsing::whitespace
                    .before(mul.or(div))
                    .followed_by(parsing::whitespace)
                    .and(pos_neg)
                    .any_amount(),
            )
            .try_parse(input)?;

        let result = terms.into_iter().fold(initial_term, |acc, (op, term)| {
            Expression::Operation(Operation::Binary(
                BinaryOperation::Arithmetic(op),
                Box::new(acc),
                Box::new(term),
            ))
        });

        Ok((result, input))
    };

    let add_sub = |input| {
        let add = '+'.map(|_| ArithmeticOperation::Add);
        let sub = '-'.map(|_| ArithmeticOperation::Sub);

        let ((initial_term, terms), input) = mul_div
            .and(
                parsing::whitespace
                    .before(add.or(sub))
                    .followed_by(parsing::whitespace)
                    .and(mul_div)
                    .any_amount(),
            )
            .try_parse(input)?;

        let result = terms.into_iter().fold(initial_term, |acc, (op, term)| {
            Expression::Operation(Operation::Binary(
                BinaryOperation::Arithmetic(op),
                Box::new(acc),
                Box::new(term),
            ))
        });

        Ok((result, input))
    };

    add_sub.try_parse(input)
}

fn primary(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    enclosed.or(atom).try_parse(input)
}

fn enclosed(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    parsing::open_paren
        .before(parsing::whitespace)
        .before(parse)
        .followed_by(parsing::whitespace)
        .followed_by(parsing::close_paren)
        .try_parse(input)
}

fn atom(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    value::parse
        .map(Expression::Literal)
        .or(identifier)
        .try_parse(input)
}

fn args(input: ParseInput<'_>) -> ParseResult<'_, UnaryOperation> {
    parsing::open_paren
        .followed_by(parsing::whitespace)
        .followed_by(parsing::close_paren)
        .map_to(UnaryOperation::Call)
        .try_parse(input)
}

fn call(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let (result, input) = primary.try_parse(input)?;
    let (calls, input) = parsing::whitespace
        .before(args)
        .any_amount()
        .try_parse(input)?;

    let result = calls.into_iter().fold(result, |acc, call| {
        Expression::Operation(Operation::Unary(call, Box::new(acc)))
    });

    Ok((result, input))
}

fn identifier(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    parsing::identifier_string
        .map(Expression::Identifier)
        .try_parse(input)
}
