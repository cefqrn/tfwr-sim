use super::{
    AssignmentTarget, Block, Closure, Context, EndReason, EndStatement, EvaluationError, Rc,
    Statement, Value,
};

impl Statement {
    pub fn execute(&self, context: &mut Context) -> Result<EndReason, EvaluationError> {
        match self {
            Self::Global(_) => Ok(EndReason::End),
            Self::Assignment(name, expr) => {
                let value = expr.evaluate(context)?;
                name.assign(context, value)?;

                Ok(EndReason::End)
            }
            Self::Call(call) => {
                call.evaluate(context)?;
                Ok(EndReason::End)
            }
            Self::If(possibilities, else_) => {
                for (condition, body) in possibilities {
                    if condition.evaluate(context)?.into() {
                        return body.evaluate(context);
                    }
                }

                else_.evaluate(context)
            }
            Self::Def(definition) => {
                let value = Value::Function(Rc::new(Closure {
                    parameters: definition.parameters.clone(),
                    body: definition.body.clone(),
                    base_context: context
                        .capture(&definition.captured)
                        .ok_or(EvaluationError)?,
                    locals: definition.locals.clone(),
                }));

                context.set(definition.name.clone(), value);

                Ok(EndReason::End)
            }
            Self::While(condition, body) => {
                while condition.evaluate(context)?.into() {
                    match body.evaluate(context)? {
                        ret @ EndReason::Return(_) => return Ok(ret),
                        EndReason::Break => break,
                        EndReason::Continue | EndReason::End => {}
                    }
                }

                Ok(EndReason::End)
            }
            Self::For(target, expr, body) => match expr.evaluate(context)? {
                Value::Tuple(values) => {
                    for value in values {
                        target.assign(context, value)?;
                        match body.evaluate(context)? {
                            ret @ EndReason::Return(_) => return Ok(ret),
                            EndReason::Break => break,
                            EndReason::Continue | EndReason::End => {}
                        }
                    }

                    Ok(EndReason::End)
                }
                Value::None
                | Value::String(_)
                | Value::Number(_)
                | Value::Bool(_)
                | Value::Function(_) => Err(EvaluationError),
            },
        }
    }
}

impl Block {
    pub fn evaluate(&self, context: &mut Context) -> Result<EndReason, EvaluationError> {
        let Self(body, end_statement) = self;
        for statement in body {
            match statement.execute(context)? {
                r @ (EndReason::Break | EndReason::Continue | EndReason::Return(_)) => {
                    return Ok(r);
                }
                EndReason::End => {}
            }
        }

        Ok(match end_statement {
            Some(EndStatement::Return(expr)) => EndReason::Return(expr.evaluate(context)?),
            Some(EndStatement::Continue) => EndReason::Continue,
            Some(EndStatement::Break) => EndReason::Break,
            None => EndReason::End,
        })
    }
}

impl AssignmentTarget {
    fn assign(&self, context: &mut Context, value: Value) -> Result<(), EvaluationError> {
        match (self, value) {
            (Self::Single(name), value) => context.set(name.clone(), value),
            (Self::Multiple(targets), Value::Tuple(values)) if targets.len() == values.len() => {
                for (name, value) in targets.iter().zip(values) {
                    name.assign(context, value)?;
                }
            }
            _ => Err(EvaluationError)?,
        }

        Ok(())
    }
}
