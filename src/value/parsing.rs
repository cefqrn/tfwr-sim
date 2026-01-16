use crate::parsing;
use parsing::{ParseError, ParseInput, ParseResult, Parser, Predicate};

use super::Value;

impl Value {
    pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, Self> {
        let none = "None".map(|_| Self::None);
        let true_ = "True".map(|_| Self::Bool(true));
        let false_ = "False".map(|_| Self::Bool(false));

        let keyword = true_
            .or(false_)
            .or(none)
            .followed_by(parsing::identifier_boundary);

        number.or(string).or(keyword).try_parse(input)
    }
}

fn number(input: ParseInput<'_>) -> ParseResult<'_, Value> {
    let digit = Predicate(&|c| c.is_ascii_digit());

    let ((whole, fractional), input) = digit
        .any_amount()
        .and('.'.before(digit.at_least_one()).maybe())
        .try_parse(input)?;

    let mut result = whole;
    if let Some((d, ds)) = fractional {
        result.push('.');
        result.push(d);
        result.extend(ds);
    } else if result.is_empty() {
        Err(ParseError)?;
    }
    let result = String::from_iter(result).parse().expect("numbers and dot");

    Ok((Value::Number(result), input))
}

fn string(input: ParseInput<'_>) -> ParseResult<'_, Value> {
    let escaped = |input| {
        let (_, rest) = '\\'.and(Predicate(&|_| true)).try_parse(input)?;
        let taken = &input.s[..input.s.len() - rest.s.len()];

        Ok((taken, rest))
    };

    let unescaped = |input| {
        let (_, rest) = Predicate(&|c| c != '"').try_parse(input)?;
        let taken = &input.s[..input.s.len() - rest.s.len()];

        Ok((taken, rest))
    };

    let (s, rest) = '"'
        .before(escaped.or(unescaped).any_amount())
        .followed_by('"')
        .try_parse(input)?;
    Ok((Value::String(s.join("")), rest))
}
