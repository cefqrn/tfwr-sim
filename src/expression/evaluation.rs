use crate::context::{Context, EvaluationError};
use crate::statement::EndReason;
use crate::value::Value;

use super::{
    ArithmeticOperation, BinaryOperation, Call, ComparisonOperation, Expression, LogicalOperation,
    Operation, UnaryOperation,
};

use std::cmp::Ordering;

impl Expression {
    pub fn evaluate(&self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Literal(v) => Ok(v.clone()),
            Self::Operation(op) => op.evaluate(context),
            Self::Identifier(n) => context.get(n).ok_or(EvaluationError),
            Self::Call(call) => call.evaluate(context),
            Self::Tuple(elements) => Ok(Value::Tuple(
                elements
                    .iter()
                    .map(|e| e.evaluate(context))
                    .collect::<Result<Vec<Value>, EvaluationError>>()?,
            )),
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

impl Call {
    pub fn evaluate(&self, context: &mut Context) -> Result<Value, EvaluationError> {
        let Self(f, args) = self;

        let Value::Function(f) = f.evaluate(context)? else {
            return Err(EvaluationError);
        };

        if args.len() != f.parameters.len() {
            return Err(EvaluationError);
        }

        let mut new_context = f.base_context.clone();
        for identifier in &f.locals {
            new_context.declare(identifier.clone());
        }

        for (name, arg) in f.parameters.iter().cloned().zip(args) {
            new_context.set(name, arg.evaluate(context)?);
        }

        f.body.evaluate(&mut new_context).map(|x| {
            if let EndReason::Return(v) = x {
                v
            } else {
                Value::None
            }
        })
    }
}
