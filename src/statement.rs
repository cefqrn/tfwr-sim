use crate::evaluation;
use crate::expression;
use crate::parsing;
use crate::value::{Closure, Value};
use evaluation::{Context, EvaluationError};
use expression::Expression;
use parsing::{ParseError, ParseInput, ParseResult, Parser};

use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Statement {
    Assignment(String, Expression),
    If(Vec<(Expression, Vec<Statement>)>, Vec<Statement>),
    Def(Definition),
    Global(String),
}

#[derive(Clone, Debug)]
pub struct Definition {
    name: String,
    parameters: Vec<String>,
    body: Vec<Statement>,
    locals: HashSet<String>,
    captured: HashSet<String>,
    captured_and_modified: HashSet<String>,
}

impl Statement {
    pub fn execute(self, context: &mut Context) -> Option<EvaluationError> {
        match self {
            Self::Assignment(name, value) => match value.evaluate(context) {
                Ok(value) => {
                    evaluation::assign(context, &name, value);
                    None
                }
                Err(e) => Some(e),
            },
            Self::If(possibilities, else_) => {
                for (condition, body) in possibilities {
                    let condition = match condition.evaluate(context) {
                        Ok(r) => r.into(),
                        Err(e) => return Some(e),
                    };

                    if condition {
                        for s in body {
                            s.execute(context);
                        }
                        return None;
                    }
                }

                for s in else_ {
                    s.execute(context);
                }

                None
            }
            Self::Def(definition) => {
                let mut base_context = Context::new();
                for name in definition
                    .captured
                    .iter()
                    .chain(&definition.captured_and_modified)
                {
                    let captured_variable = evaluation::capture(context, name);
                    evaluation::add(&mut base_context, name.to_owned(), captured_variable);
                }

                let value = Value::Function(Rc::new(Closure {
                    parameters: definition.parameters,
                    body: definition.body,
                    base_context,
                    locals: definition.locals.clone(),
                }));

                evaluation::assign(context, definition.name.as_str(), value);

                None
            }
            Self::Global(_) => None,
        }
    }
}

pub fn statement(input: ParseInput<'_>) -> ParseResult<'_, Statement> {
    let assignment = parsing::assignable
        .followed_by(parsing::spaces)
        .followed_by('=')
        .followed_by(parsing::spaces)
        .and(expression::parse)
        .map(|(name, value)| Statement::Assignment(name, value));

    let global = "global"
        .before(parsing::spaces)
        .before(parsing::assignable)
        .map(Statement::Global);

    let single_line = assignment
        .or(global)
        .followed_by(parsing::up_to_next_statement);

    let if_ = "if"
        .before(parsing::spaces)
        .before(expression::parse)
        .followed_by(parsing::spaces)
        .followed_by(':')
        .followed_by(parsing::up_to_next_statement)
        .and(block)
        .and(
            input
                .indentation
                .before("elif")
                .before(parsing::spaces)
                .before(expression::parse)
                .followed_by(parsing::spaces)
                .followed_by(':')
                .followed_by(parsing::up_to_next_statement)
                .and(block)
                .any_amount(),
        )
        .and(
            input
                .indentation
                .before("else")
                .before(parsing::spaces)
                .before(':')
                .followed_by(parsing::up_to_next_statement)
                .before(block)
                .maybe(),
        )
        .map(|((initial, elifs), else_)| {
            let mut possibilities = vec![initial];
            possibilities.extend(elifs);

            Statement::If(possibilities, else_.unwrap_or_else(Vec::new))
        });

    let multi_line = if_.or(def);

    input
        .indentation
        .before(multi_line.or(single_line))
        .try_parse(input)
}

pub fn block(input: ParseInput<'_>) -> ParseResult<'_, Vec<Statement>> {
    // look ahead to get current indentation
    let (indentation, _) = parsing::spaces.try_parse(input)?;
    let indentation_length = indentation.into_iter().map(char::len_utf8).sum();
    let indentation = &input.s[..indentation_length];

    // ensure indentation increased and is consistent
    if indentation
        .strip_prefix(input.indentation)
        .into_iter()
        .all(str::is_empty)
    {
        Err(ParseError)?;
    }

    // set indentation
    let initial_indentation = input.indentation;
    let input = ParseInput {
        indentation,
        ..input
    };

    let (result, input) = statement.any_amount().try_parse(input)?;

    // reset indentation
    let input = ParseInput {
        indentation: initial_indentation,
        ..input
    };

    Ok((result, input))
}

pub fn module(input: ParseInput<'_>) -> ParseResult<'_, (Context, Vec<Statement>)> {
    let (body, input) = statement.any_amount().try_parse(input)?;
    if !input.fully_consumed {
        Err(ParseError)?;
    }

    let mut locals = HashSet::new();
    let mut used_before_assignment = HashSet::new();
    let mut unnecessarily_marked_global = HashSet::new();
    for s in &body {
        classify_vars(
            s,
            &mut locals,
            &mut used_before_assignment,
            &mut unnecessarily_marked_global,
        )?;
    }

    if !used_before_assignment.is_empty() {
        Err(ParseError)?;
    }

    let mut context = Context::new();
    for var in locals
        .into_iter()
        .chain(unnecessarily_marked_global.into_iter())
    {
        evaluation::declare(&mut context, var);
    }

    Ok(((context, body), input))
}

fn def(input: ParseInput<'_>) -> ParseResult<'_, Statement> {
    let (((name, parameters), body), input) = "def"
        .before(parsing::spaces)
        .before(parsing::assignable)
        .followed_by(parsing::spaces)
        .followed_by(parsing::open_paren)
        .and(
            parsing::whitespace
                .before(parsing::assignable)
                .followed_by(parsing::whitespace.before(',').maybe())
                .any_amount(),
        )
        .followed_by(parsing::whitespace)
        .followed_by(parsing::close_paren)
        .followed_by(parsing::spaces)
        .followed_by(':')
        .followed_by(parsing::up_to_next_statement)
        .and(block)
        .try_parse(input)?;

    let mut locals = HashSet::new();
    for var in &parameters {
        locals.insert(var.to_owned());
    }

    let mut captured = HashSet::new();
    let mut captured_and_modified = HashSet::new();
    for s in &body {
        classify_vars(s, &mut locals, &mut captured, &mut captured_and_modified)?;
    }

    Ok((
        Statement::Def(Definition {
            name,
            parameters,
            body,
            locals,
            captured,
            captured_and_modified,
        }),
        input,
    ))
}

fn classify_vars(
    statement: &Statement,
    locals: &mut HashSet<String>,
    captured: &mut HashSet<String>,
    captured_and_modified: &mut HashSet<String>,
) -> Result<(), ParseError> {
    fn see(
        var: String,
        locals: &HashSet<String>,
        captured: &mut HashSet<String>,
        captured_and_modified: &HashSet<String>,
    ) {
        if !locals.contains(&var) && !captured_and_modified.contains(&var) {
            captured.insert(var);
        }
    }

    fn assign(
        var: String,
        locals: &mut HashSet<String>,
        captured: &HashSet<String>,
        captured_and_modified: &HashSet<String>,
    ) -> Result<(), ParseError> {
        if captured.contains(&var) {
            return Err(ParseError);
        }

        if !captured_and_modified.contains(&var) {
            locals.insert(var);
        }

        Ok(())
    }

    fn global(
        var: String,
        locals: &HashSet<String>,
        captured: &mut HashSet<String>,
        captured_and_modified: &mut HashSet<String>,
    ) -> Result<(), ParseError> {
        if locals.contains(&var) {
            return Err(ParseError);
        }

        captured.remove(&var);
        captured_and_modified.insert(var);

        Ok(())
    }

    match statement {
        Statement::Assignment(name, expr) => {
            for var in expr.identifiers() {
                see(var, locals, captured, captured_and_modified);
            }

            assign(name.to_owned(), locals, captured, captured_and_modified)?;
        }
        Statement::Def(definition) => {
            // assign name first to allow for recursion
            assign(
                definition.name.clone(),
                locals,
                captured,
                captured_and_modified,
            )?;

            // assume all assignments and observations inside the function
            // happen at the definition to avoid having to know
            // when the function is called
            for var in &definition.captured {
                see(var.to_owned(), locals, captured, captured_and_modified);
            }
            for var in &definition.captured_and_modified {
                assign(var.to_owned(), locals, captured, captured_and_modified)?;
            }
        }
        Statement::Global(name) => {
            global(name.to_owned(), locals, captured, captured_and_modified)?;
        }
        Statement::If(possibilities, else_) => {
            for (condition, body) in possibilities {
                for var in condition.identifiers() {
                    see(var, locals, captured, captured_and_modified);
                }

                for s in body {
                    classify_vars(s, locals, captured, captured_and_modified)?;
                }
            }

            for s in else_ {
                classify_vars(s, locals, captured, captured_and_modified)?;
            }
        }
    }

    Ok(())
}
