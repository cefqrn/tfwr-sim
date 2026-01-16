use crate::expression::{Call, Expression, Operation};
use crate::parsing::ParseError;
use crate::statement::{AssignmentTarget, Block, EndStatement, Statement};

use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, Default)]
pub enum VariableState {
    Global,
    Local,
    #[default]
    Unmarked,
}

impl VariableState {
    pub const fn after_assignment(self) -> Self {
        match self {
            Self::Global => Self::Global,
            Self::Local | Self::Unmarked => Self::Local,
        }
    }

    pub const fn after_global(self) -> Result<Self, ParseError> {
        match self {
            Self::Global | Self::Unmarked => Ok(Self::Global),
            Self::Local => Err(ParseError),
        }
    }
}

pub fn classify_vars(
    statement: &Statement,
    variables: &mut HashMap<String, VariableState>,
) -> Result<(), ParseError> {
    match statement {
        Statement::Assignment(target, expr) => {
            for identifier in expr.identifiers() {
                variables.entry(identifier).or_default();
            }

            target.mark_assigned(variables);
        }
        Statement::Call(call) => {
            for identifier in call.identifiers() {
                variables.entry(identifier).or_default();
            }
        }
        Statement::Def(definition) => {
            variables
                .entry(definition.name.clone())
                .and_modify(|state| *state = state.after_assignment())
                .or_insert(VariableState::Local);

            for identifier in definition.captured.iter().cloned() {
                variables.entry(identifier).or_default();
            }
        }
        Statement::Global(identifier) => {
            let entry = variables.entry(identifier.clone()).or_default();
            *entry = entry.after_global()?;
        }
        Statement::If(possibilities, else_) => {
            for (condition, body) in possibilities {
                for identifier in condition.identifiers() {
                    variables.entry(identifier).or_default();
                }

                body.classify_vars(variables)?;
            }

            else_.classify_vars(variables)?;
        }
        Statement::While(condition, body) => {
            for identifier in condition.identifiers() {
                variables.entry(identifier).or_default();
            }

            body.classify_vars(variables)?;
        }
        Statement::For(target, expr, body) => {
            for identifier in expr.identifiers() {
                variables.entry(identifier).or_default();
            }

            target.mark_assigned(variables);

            body.classify_vars(variables)?;
        }
    }

    Ok(())
}

impl Block {
    pub fn classify_vars(
        &self,
        variables: &mut HashMap<String, VariableState>,
    ) -> Result<(), ParseError> {
        let Self(body, end_statement) = self;

        for s in body {
            classify_vars(s, variables)?;
        }

        if let Some(EndStatement::Return(expr)) = end_statement {
            for identifier in expr.identifiers() {
                variables.entry(identifier).or_default();
            }
        }

        Ok(())
    }
}

impl Expression {
    #[must_use]
    fn identifiers(&self) -> HashSet<String> {
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

impl Call {
    #[must_use]
    fn identifiers(&self) -> HashSet<String> {
        let Self(f, args) = self;

        let mut result = HashSet::new();
        f.identifiers_inner(&mut result);
        for arg in args {
            arg.identifiers_inner(&mut result);
        }

        result
    }
}

impl AssignmentTarget {
    fn mark_assigned(&self, variables: &mut HashMap<String, VariableState>) {
        match self {
            Self::Single(identifier) => {
                variables
                    .entry(identifier.clone())
                    .and_modify(|state| {
                        *state = state.after_assignment();
                    })
                    .or_insert_with(|| VariableState::Local);
            }
            Self::Multiple(targets) => {
                for target in targets {
                    target.mark_assigned(variables);
                }
            }
        }
    }
}
