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
    Assignment(AssignmentTarget, Expression),
    Global(String),
    Def(Definition),
    If(Vec<(Expression, Block)>, Block),
    While(Expression, Block),
}

#[derive(Clone, Debug)]
pub struct Definition {
    name: String,
    parameters: Vec<String>,
    body: Block,
    locals: HashSet<String>,
    captured: HashSet<String>,
    captured_and_modified: HashSet<String>,
}

#[derive(Clone, Debug)]
pub enum AssignmentTarget {
    Single(String),
    Multiple(Vec<AssignmentTarget>),
}

#[derive(Clone, Debug)]
pub struct Block(Vec<Statement>, Option<EndStatement>);

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

impl Statement {
    pub fn execute(&self, context: &mut Context) -> Result<EndReason, EvaluationError> {
        match self {
            Self::Global(_) => Ok(EndReason::End),
            Self::Assignment(name, expr) => {
                let value = expr.evaluate(context)?;
                name.assign(context, value)?;

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
                    parameters: definition.parameters.clone(),
                    body: definition.body.clone(),
                    base_context,
                    locals: definition.locals.clone(),
                }));

                evaluation::assign(context, definition.name.as_str(), value);

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

    #[must_use]
    pub const fn empty() -> Self {
        Self(Vec::new(), None)
    }

    fn classify_vars(
        &self,
        locals: &mut HashSet<String>,
        captured: &mut HashSet<String>,
        captured_and_modified: &mut HashSet<String>,
    ) -> Result<(), ParseError> {
        let Self(body, end_statement) = self;

        for s in body {
            classify_vars(s, locals, captured, captured_and_modified)?;
        }

        if let Some(EndStatement::Return(expr)) = end_statement {
            for var in expr.identifiers() {
                see(var, locals, captured, captured_and_modified);
            }
        }

        Ok(())
    }
}

impl AssignmentTarget {
    fn assign(&self, context: &mut Context, value: Value) -> Result<(), EvaluationError> {
        match (self, value) {
            (Self::Single(name), value) => evaluation::assign(context, name, value),
            (Self::Multiple(targets), Value::Tuple(values)) if targets.len() == values.len() => {
                for (name, value) in targets.iter().zip(values) {
                    name.assign(context, value)?;
                }
            }
            _ => Err(EvaluationError)?,
        }

        Ok(())
    }

    fn parse(input: ParseInput<'_>) -> ParseResult<'_, Self> {
        fn element(input: ParseInput<'_>) -> ParseResult<'_, AssignmentTarget> {
            parsing::open_paren
                .before(parsing::whitespace)
                .before(
                    AssignmentTarget::parse
                        .followed_by(parsing::whitespace)
                        .or(parsing::nothing.map(|()| AssignmentTarget::Multiple(Vec::new()))),
                )
                .followed_by(parsing::close_paren)
                .or(parsing::assignable.map(AssignmentTarget::Single))
                .try_parse(input)
        }

        element
            .followed_by(parsing::whitespace)
            .followed_by(',')
            .and(
                parsing::whitespace
                    .before(element)
                    .followed_by(parsing::whitespace.followed_by(',').maybe())
                    .any_amount(),
            )
            .map(|(fst, rest)| {
                let mut result = vec![fst];
                result.extend(rest);
                Self::Multiple(result)
            })
            .or(element)
            .try_parse(input)
    }
}

pub fn statement(input: ParseInput<'_>) -> ParseResult<'_, Statement> {
    let assignment = AssignmentTarget::parse
        .followed_by(parsing::spaces)
        .followed_by('=')
        .followed_by(parsing::spaces)
        .and(expression::parse)
        .map(|(target, value)| Statement::Assignment(target, value));

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

            Statement::If(possibilities, else_.unwrap_or_else(Block::empty))
        });

    let while_ = "while"
        .before(parsing::spaces)
        .before(expression::parse)
        .followed_by(parsing::spaces)
        .followed_by(':')
        .followed_by(parsing::up_to_next_statement)
        .and(block)
        .map(|(condition, body)| Statement::While(condition, body));

    let multi_line = if_.or(def).or(while_);

    input
        .indentation
        .before(multi_line.or(single_line))
        .try_parse(input)
}

pub fn block(input: ParseInput<'_>) -> ParseResult<'_, Block> {
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

    let ((body, end), input) = statement
        .any_amount()
        .and(
            indentation
                .before(
                    "continue"
                        .map(|_| EndStatement::Continue)
                        .or("break".map(|_| EndStatement::Break))
                        .or("return"
                            .before(parsing::identifier_boundary)
                            .before(parsing::spaces)
                            .before(expression::parse)
                            .map(EndStatement::Return)),
                )
                .followed_by(parsing::up_to_next_statement)
                .and(statement.maybe())
                .maybe(),
        )
        .try_parse(input)?;

    // reset indentation
    let input = ParseInput {
        indentation: initial_indentation,
        ..input
    };

    if let Some((end_statement, statement_after_end)) = end {
        if statement_after_end.is_some() {
            Err(ParseError)
        } else {
            Ok((Block(body, Some(end_statement)), input))
        }
    } else {
        Ok((Block(body, None), input))
    }
}

pub fn module(input: ParseInput<'_>) -> ParseResult<'_, (Context, Block)> {
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

    Ok(((context, Block(body, None)), input))
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

    body.classify_vars(&mut locals, &mut captured, &mut captured_and_modified)?;

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

fn classify_vars(
    statement: &Statement,
    locals: &mut HashSet<String>,
    captured: &mut HashSet<String>,
    captured_and_modified: &mut HashSet<String>,
) -> Result<(), ParseError> {
    match statement {
        Statement::Assignment(target, expr) => {
            fn assign_target(
                target: &AssignmentTarget,
                locals: &mut HashSet<String>,
                captured: &mut HashSet<String>,
                captured_and_modified: &mut HashSet<String>,
            ) -> Result<(), ParseError> {
                match target {
                    AssignmentTarget::Single(name) => {
                        assign(name.to_owned(), locals, captured, captured_and_modified)
                    }
                    AssignmentTarget::Multiple(targets) => {
                        for target in targets {
                            assign_target(target, locals, captured, captured_and_modified)?;
                        }

                        Ok(())
                    }
                }
            }

            for var in expr.identifiers() {
                see(var, locals, captured, captured_and_modified);
            }

            assign_target(target, locals, captured, captured_and_modified)?;
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

                body.classify_vars(locals, captured, captured_and_modified)?;
            }

            else_.classify_vars(locals, captured, captured_and_modified)?;
        }
        Statement::While(condition, body) => {
            for var in condition.identifiers() {
                see(var, locals, captured, captured_and_modified);
            }

            body.classify_vars(locals, captured, captured_and_modified)?;
        }
    }

    Ok(())
}
