use crate::expression::Expression;
use crate::parsing;
use parsing::{ParseError, ParseInput, ParseResult, Parser};

use super::{
    AssignmentTarget, Block, Call, Definition, EndStatement, HashMap, Statement, VariableState,
};

impl Statement {
    pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, Self> {
        let assignment = AssignmentTarget::parse
            .followed_by(parsing::spaces)
            .followed_by('=')
            .followed_by(parsing::spaces)
            .and(Expression::parse)
            .map(|(target, value)| Self::Assignment(target, value));

        let global = "global"
            .before(parsing::spaces)
            .before(parsing::assignable)
            .map(Self::Global);

        let single_line = assignment
            .or(global)
            .or(Call::parse.map(Self::Call))
            .followed_by(parsing::up_to_next_statement);

        let if_ = "if"
            .before(parsing::spaces)
            .before(Expression::parse)
            .followed_by(parsing::spaces)
            .followed_by(':')
            .followed_by(parsing::up_to_next_statement)
            .and(Block::parse)
            .and(
                input
                    .indentation
                    .before("elif")
                    .before(parsing::spaces)
                    .before(Expression::parse)
                    .followed_by(parsing::spaces)
                    .followed_by(':')
                    .followed_by(parsing::up_to_next_statement)
                    .and(Block::parse)
                    .any_amount(),
            )
            .and(
                input
                    .indentation
                    .before("else")
                    .before(parsing::spaces)
                    .before(':')
                    .followed_by(parsing::up_to_next_statement)
                    .before(Block::parse)
                    .maybe(),
            )
            .map(|((initial, elifs), else_)| {
                let mut possibilities = vec![initial];
                possibilities.extend(elifs);

                Self::If(possibilities, else_.unwrap_or_default())
            });

        let while_ = "while"
            .before(parsing::spaces)
            .before(Expression::parse)
            .followed_by(parsing::spaces)
            .followed_by(':')
            .followed_by(parsing::up_to_next_statement)
            .and(Block::parse)
            .map(|(condition, body)| Self::While(condition, body));

        let for_ = "for"
            .before(parsing::spaces)
            .before(AssignmentTarget::parse)
            .followed_by(parsing::spaces)
            .followed_by("in")
            .followed_by(parsing::spaces)
            .and(Expression::parse)
            .followed_by(parsing::spaces)
            .followed_by(':')
            .followed_by(parsing::up_to_next_statement)
            .and(Block::parse)
            .map(|((target, expr), body)| Self::For(target, expr, body));

        let multi_line = if_.or(def).or(while_).or(for_);

        input
            .indentation
            .before(multi_line.or(single_line))
            .try_parse(input)
    }
}

impl Block {
    pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, Self> {
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

        let ((body, end), input) = Statement::parse
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
                                .before(Expression::parse)
                                .map(EndStatement::Return)),
                    )
                    .followed_by(parsing::up_to_next_statement)
                    .and(Statement::parse.maybe())
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
                Ok((Self(body, Some(end_statement)), input))
            }
        } else {
            Ok((Self(body, None), input))
        }
    }
}

impl AssignmentTarget {
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
        .and(Block::parse)
        .try_parse(input)?;

    // start parameters off as locals
    let mut variables = HashMap::new();
    variables.extend(
        parameters
            .iter()
            .map(|identifier| (identifier.clone(), VariableState::Local)),
    );

    body.classify_vars(&mut variables)?;

    let mut locals = Vec::new();
    let mut captured = Vec::new();
    for (identifier, state) in variables {
        match state {
            VariableState::Global | VariableState::Unmarked => &mut captured,
            VariableState::Local => &mut locals,
        }
        .push(identifier);
    }

    Ok((
        Statement::Def(Definition {
            name,
            parameters,
            body,
            locals,
            captured,
        }),
        input,
    ))
}
