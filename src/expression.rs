use crate::evaluation;
use crate::parsing;
use crate::value;
use evaluation::{Context, EvaluationError};
use parsing::{ParseInput, ParseResult, Parser};
use value::Value;

use std::cmp::Ordering;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub enum Expression {
    Literal(Value),
    Identifier(String),
    Operation(Operation),
    Tuple(Vec<Expression>),
    Call(Call),
}

#[derive(Clone, Debug)]
pub struct Call(pub Box<Expression>, pub Vec<Expression>);

#[derive(Clone, Debug)]
pub enum Operation {
    Unary(UnaryOperation, Box<Expression>),
    Binary(BinaryOperation, Box<Expression>, Box<Expression>),
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOperation {
    Pos,
    Neg,
}

#[derive(Clone, Copy, Debug)]
pub enum BinaryOperation {
    Arithmetic(ArithmeticOperation),
    Logical(LogicalOperation),
    Comparison(ComparisonOperation),
}

#[derive(Clone, Copy, Debug)]
pub enum ArithmeticOperation {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    FloorDiv,
    Exp,
}

#[derive(Clone, Copy, Debug)]
pub enum LogicalOperation {
    And,
    Or,
}

#[derive(Clone, Copy, Debug)]
pub enum ComparisonOperation {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    Ne,
}

impl Expression {
    pub fn evaluate(&self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Literal(v) => Ok(v.clone()),
            Self::Operation(op) => op.evaluate(context),
            Self::Identifier(n) => context.get(n).map_or_else(
                || Err(EvaluationError),
                |v| v.borrow().clone().ok_or(EvaluationError),
            ),
            Self::Call(Call(f, args)) => {
                let Value::Function(f) = f.evaluate(context)? else {
                    return Err(EvaluationError);
                };

                if args.len() != f.parameters.len() {
                    return Err(EvaluationError);
                }

                let mut new_context = f.base_context.clone();
                for name in &f.locals {
                    evaluation::declare(&mut new_context, name.clone());
                }
                for (name, arg) in f.parameters.iter().zip(args) {
                    evaluation::assign(&mut new_context, name, arg.evaluate(context)?);
                }

                for s in &f.body {
                    s.execute(&mut new_context);
                }

                Ok(Value::None)
            }
            Self::Tuple(elements) => Ok(Value::Tuple(
                elements
                    .iter()
                    .map(|e| e.evaluate(context))
                    .collect::<Result<Vec<Value>, EvaluationError>>()?,
            )),
        }
    }

    #[must_use]
    pub fn identifiers(&self) -> HashSet<String> {
        let mut result = HashSet::new();
        self.identifiers_inner(&mut result);

        result
    }

    fn identifiers_inner(&self, result: &mut HashSet<String>) {
        match self {
            Self::Literal(_) => {}
            Self::Identifier(name) => {
                result.insert(name.to_owned());
            }
            Self::Operation(operation) => match operation {
                Operation::Unary(_, x) => {
                    x.identifiers_inner(result);
                }
                Operation::Binary(_, x, y) => {
                    x.identifiers_inner(result);
                    y.identifiers_inner(result);
                }
            },
            Self::Tuple(expressions) => {
                for e in expressions {
                    e.identifiers_inner(result);
                }
            }
            Self::Call(Call(f, args)) => {
                f.identifiers_inner(result);
                for e in args {
                    e.identifiers_inner(result);
                }
            }
        }
    }
}

impl Operation {
    pub fn evaluate(&self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Unary(op, x) => {
                let x = x.evaluate(context)?;
                match op {
                    UnaryOperation::Pos => match x {
                        Value::Number(_) | Value::Bool(_) => Ok(x),
                        _ => Err(EvaluationError),
                    },
                    UnaryOperation::Neg => f64::try_from(x).map(|n| Value::Number(-n)),
                }
            }
            Self::Binary(op, x, y) => {
                let x = x.evaluate(context)?;

                match op {
                    BinaryOperation::Arithmetic(op) => {
                        let x: f64 = x.try_into()?;
                        let y: f64 = y.evaluate(context)?.try_into()?;

                        match op {
                            ArithmeticOperation::Add => Ok(Value::Number(x + y)),
                            ArithmeticOperation::Sub => Ok(Value::Number(x - y)),
                            ArithmeticOperation::Mul => Ok(Value::Number(x * y)),
                            ArithmeticOperation::Div
                            | ArithmeticOperation::Mod
                            | ArithmeticOperation::FloorDiv
                                if y == 0. =>
                            {
                                Err(EvaluationError)
                            }
                            ArithmeticOperation::Div => Ok(Value::Number(x / y)),
                            ArithmeticOperation::Mod => Ok(Value::Number(x.rem_euclid(y))),
                            ArithmeticOperation::FloorDiv => Ok(Value::Number(x.div_euclid(y))),
                            ArithmeticOperation::Exp => Ok(Value::Number(x.powf(y))),
                        }
                    }
                    BinaryOperation::Logical(op) => match op {
                        LogicalOperation::And => {
                            if (&x).into() {
                                y.evaluate(context)
                            } else {
                                Ok(x)
                            }
                        }
                        LogicalOperation::Or => {
                            if (&x).into() {
                                Ok(x)
                            } else {
                                y.evaluate(context)
                            }
                        }
                    },
                    BinaryOperation::Comparison(op) => {
                        let y = y.evaluate(context)?;
                        match (op, x.partial_cmp(&y)) {
                            (
                                ComparisonOperation::Eq
                                | ComparisonOperation::Ge
                                | ComparisonOperation::Le,
                                Some(Ordering::Equal),
                            )
                            | (
                                ComparisonOperation::Ge
                                | ComparisonOperation::Gt
                                | ComparisonOperation::Ne,
                                Some(Ordering::Greater),
                            )
                            | (
                                ComparisonOperation::Le
                                | ComparisonOperation::Lt
                                | ComparisonOperation::Ne,
                                Some(Ordering::Less),
                            )
                            | (ComparisonOperation::Ne, None) => Ok(Value::Bool(true)),
                            (_, Some(_)) | (ComparisonOperation::Eq, None) => {
                                Ok(Value::Bool(false))
                            }
                            (_, None) => Err(EvaluationError),
                        }
                    }
                }
            }
        }
    }
}

pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let tuple = {
        element
            .followed_by(parsing::whitespace)
            .followed_by(',')
            .and(
                parsing::whitespace
                    .before(parse)
                    .followed_by(parsing::whitespace.followed_by(',').maybe())
                    .any_amount(),
            )
            .map(|(fst, rest)| {
                let mut result = vec![fst];
                result.extend(rest);
                Expression::Tuple(result)
            })
            .or(element)
    };

    tuple.try_parse(input)
}

fn element(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let mul_div = {
        let mul = '*'.map_to(ArithmeticOperation::Mul);
        let div = '/'.map_to(ArithmeticOperation::Div);
        let mod_ = '%'.map_to(ArithmeticOperation::Mod);
        let fdiv = "//".map_to(ArithmeticOperation::FloorDiv);
        binop(
            pos_neg(exp),
            mul.or(fdiv)
                .or(div)
                .or(mod_)
                .map(BinaryOperation::Arithmetic),
        )
    };

    let add_sub = {
        let add = '+'.map_to(ArithmeticOperation::Add);
        let sub = '-'.map_to(ArithmeticOperation::Sub);
        binop(mul_div, add.or(sub).map(BinaryOperation::Arithmetic))
    };

    let cmp = {
        let eq = "==".map_to(ComparisonOperation::Eq);
        let ge = ">=".map_to(ComparisonOperation::Ge);
        let gt = ">".map_to(ComparisonOperation::Gt);
        let le = "<=".map_to(ComparisonOperation::Le);
        let lt = "<".map_to(ComparisonOperation::Lt);
        let ne = "!=".map_to(ComparisonOperation::Ne);

        let op = eq.or(gt).or(ge).or(lt).or(le).or(ne);

        // no chaining
        add_sub
            .followed_by(parsing::whitespace)
            .and(op)
            .followed_by(parsing::whitespace)
            .and(add_sub)
            .map(|((x, op), y)| {
                Expression::Operation(Operation::Binary(
                    BinaryOperation::Comparison(op),
                    Box::new(x),
                    Box::new(y),
                ))
            })
            .or(add_sub)
    };

    let and = {
        let op = "and"
            .followed_by(parsing::identifier_boundary)
            .map_to(LogicalOperation::And);
        binop(cmp, op.map(BinaryOperation::Logical))
    };

    let or = {
        let op = "or"
            .followed_by(parsing::identifier_boundary)
            .map_to(LogicalOperation::Or);
        binop(and, op.map(BinaryOperation::Logical))
    };

    or.try_parse(input)
}

fn binop<'a>(
    atom: impl Parser<'a, Expression>,
    operation: impl Parser<'a, BinaryOperation>,
) -> impl Parser<'a, Expression> {
    atom.and(
        parsing::whitespace
            .before(operation)
            .followed_by(parsing::whitespace)
            .and(atom)
            .any_amount(),
    )
    .map(|(initial_term, terms)| {
        terms.into_iter().fold(initial_term, |acc, (op, term)| {
            Expression::Operation(Operation::Binary(op, Box::new(acc), Box::new(term)))
        })
    })
}

fn pos_neg<'a>(atom: impl Parser<'a, Expression>) -> impl Parser<'a, Expression> {
    let pos = '+'.map_to(UnaryOperation::Pos);
    let neg = '-'.map_to(UnaryOperation::Neg);

    neg.or(pos)
        .followed_by(parsing::whitespace)
        .any_amount()
        .and(atom)
        .map(|(ops, base)| {
            ops.into_iter().rev().fold(base, |acc, op| {
                Expression::Operation(Operation::Unary(op, Box::new(acc)))
            })
        })
}

fn exp(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    // right associative
    call.followed_by(parsing::whitespace)
        .followed_by("**")
        .followed_by(parsing::whitespace)
        .and(pos_neg(exp))
        .map(|(x, y)| {
            Expression::Operation(Operation::Binary(
                BinaryOperation::Arithmetic(ArithmeticOperation::Exp),
                Box::new(x),
                Box::new(y),
            ))
        })
        .or(pos_neg(call))
        .try_parse(input)
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

fn args(input: ParseInput<'_>) -> ParseResult<'_, Vec<Expression>> {
    parsing::open_paren
        .before(
            parsing::whitespace
                .before(element)
                .followed_by(parsing::whitespace.followed_by(',').maybe())
                .any_amount(),
        )
        .followed_by(parsing::whitespace)
        .followed_by(parsing::close_paren)
        .try_parse(input)
}

fn call(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let (result, input) = primary.try_parse(input)?;
    let (calls, input) = parsing::whitespace
        .before(args)
        .any_amount()
        .try_parse(input)?;

    let result = calls.into_iter().fold(result, |acc, args| {
        Expression::Call(Call(Box::new(acc), args))
    });

    Ok((result, input))
}

fn identifier(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    parsing::identifier_string
        .map(Expression::Identifier)
        .try_parse(input)
}
