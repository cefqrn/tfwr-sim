use crate::evaluation;
use crate::expression;
use crate::parsing;
use crate::value::Value;
use evaluation::{Context, EvaluationError};
use expression::Expression;
use parsing::{ParseError, ParseInput, ParseResult, Parser};

#[derive(Clone, Debug)]
pub enum Statement {
    Assignment(String, Expression),
    If(Vec<(Expression, Vec<Statement>)>, Vec<Statement>),
    Def(String, Vec<String>, Vec<Statement>),
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
                    let condition = match condition
                        .evaluate(context)
                        .map(|c| c.as_bool().expect("as_bool shouldn't error"))
                    {
                        Ok(Value::Bool(condition)) => condition,
                        Ok(_) => unreachable!(),
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
            Self::Def(name, parameters, body) => {
                let value = Value::Function(parameters, body);
                evaluation::assign(context, &name, value);
                None
            }
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

    let single_line = assignment.followed_by(parsing::up_to_next_statement);

    let if_ = "if"
        .before(parsing::spaces)
        .before(expression::parse)
        .followed_by(parsing::spaces)
        .followed_by(':')
        .followed_by(parsing::up_to_next_statement)
        .and(block)
        .and(
            "elif"
                .before(parsing::spaces)
                .before(expression::parse)
                .followed_by(parsing::spaces)
                .followed_by(':')
                .followed_by(parsing::up_to_next_statement)
                .and(block)
                .any_amount(),
        )
        .and(
            "else"
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

    let function = "def"
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
        .map(|((name, params), body)| Statement::Def(name, params, body));

    let multi_line = if_.or(function);

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
    let (result, input) = statement.any_amount().try_parse(input)?;
    if !input.fully_consumed {
        Err(ParseError)?;
    }

    let mut context = Context::new();
    for var in evaluation::assigned_to_in(&result) {
        context.insert(var.to_owned(), (None, Vec::new()));
    }

    Ok(((context, result), input))
}
