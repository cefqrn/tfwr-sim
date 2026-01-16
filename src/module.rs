use crate::context::Context;
use crate::parsing::{ParseError, ParseInput, ParseResult, Parser};
use crate::statement::{Block, Statement};
use crate::variable;

use std::collections::HashMap;

pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, (Context, Block)> {
    let (body, input) = Statement::parse.any_amount().try_parse(input)?;
    if !input.fully_consumed {
        Err(ParseError)?;
    }

    let mut variables = HashMap::new();
    for s in &body {
        variable::classify_vars(s, &mut variables)?;
    }

    // start everything off as a local and error at runtime
    let mut context = Context::new();
    for (identifier, _) in variables {
        context.declare(identifier);
    }

    Ok(((context, Block(body, None)), input))
}
