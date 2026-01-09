#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::collections::HashMap;

#[derive(Debug)]
pub struct ParseError;

#[derive(Clone, Copy, Debug)]
pub struct ParseInput<'a> {
    enclosure_amount: usize,
    in_global_scope: bool,
    indentation: &'a str,
    s: &'a str,
}

impl<'a> From<&'a str> for ParseInput<'a> {
    fn from(value: &'a str) -> Self {
        ParseInput {
            enclosure_amount: 0,
            in_global_scope: true,
            indentation: "",
            s: value.trim(),
        }
    }
}

pub type ParseResult<'a, T> = Result<(T, ParseInput<'a>), ParseError>;

pub trait Parser<'a, T>
where
    Self: Sized,
{
    fn try_parse(&self, input: ParseInput<'a>) -> ParseResult<'a, T>;

    fn maybe(self) -> impl Parser<'a, Option<T>> {
        move |input| match self.try_parse(input) {
            Ok((x, input)) => Ok((Some(x), input)),
            Err(_) => Ok((None, input)),
        }
    }

    fn any_amount(self) -> impl Parser<'a, Vec<T>> {
        move |mut input| {
            let mut result = Vec::new();
            while let Ok((x, rest)) = self.try_parse(input) {
                result.push(x);
                input = rest;
            }

            Ok((result, input))
        }
    }

    fn at_least_one(self) -> impl Parser<'a, (T, Vec<T>)> {
        move |input| {
            let (x, mut input) = self.try_parse(input)?;

            let mut xs = Vec::new();
            while let Ok((x, rest)) = self.try_parse(input) {
                xs.push(x);
                input = rest;
            }

            Ok(((x, xs), input))
        }
    }

    fn followed_by<U>(self, other: impl Parser<'a, U>) -> impl Parser<'a, T> {
        move |input| {
            let (x, input) = self.try_parse(input)?;
            let (_, input) = other.try_parse(input)?;

            Ok((x, input))
        }
    }

    fn before<U>(self, other: impl Parser<'a, U>) -> impl Parser<'a, U> {
        move |input| {
            let (_, input) = self.try_parse(input)?;
            let (x, input) = other.try_parse(input)?;

            Ok((x, input))
        }
    }

    fn and<U>(self, other: impl Parser<'a, U>) -> impl Parser<'a, (T, U)> {
        move |input| {
            let (x, input) = self.try_parse(input)?;
            let (y, input) = other.try_parse(input)?;

            Ok(((x, y), input))
        }
    }

    fn or(self, other: impl Parser<'a, T>) -> impl Parser<'a, T> {
        move |input| match self.try_parse(input) {
            r @ Ok(_) => r,
            Err(_) => other.try_parse(input),
        }
    }

    fn lookahead<U>(self, other: impl Parser<'a, U>) -> impl Parser<'a, T> {
        move |input| {
            let (x, input) = self.try_parse(input)?;
            let _ = other.try_parse(input)?;

            Ok((x, input))
        }
    }

    fn map<U>(self, f: impl Fn(T) -> U) -> impl Parser<'a, U> {
        move |input| match self.try_parse(input) {
            Ok((x, input)) => Ok((f(x), input)),
            Err(e) => Err(e),
        }
    }
}

impl<'b, T, U: Fn(ParseInput<'b>) -> ParseResult<'b, T>> Parser<'b, T> for U {
    fn try_parse(&self, input: ParseInput<'b>) -> ParseResult<'b, T> {
        self(input)
    }
}

// Pattern is nightly
// and Fn(char) -> bool conflicts with above
#[derive(Clone, Copy)]
struct Predicate<'a>(&'a dyn Fn(char) -> bool);
impl<'b> Parser<'b, char> for Predicate<'_> {
    fn try_parse(&self, input: ParseInput<'b>) -> ParseResult<'b, char> {
        let mut chars = input.s.chars();
        let c = chars.next().ok_or(ParseError)?;
        self.0(c)
            .then_some((
                c,
                ParseInput {
                    s: chars.as_str(),
                    ..input
                },
            ))
            .ok_or(ParseError)
    }
}

impl Parser<'_, Self> for &str {
    fn try_parse<'a>(&self, input: ParseInput<'a>) -> ParseResult<'a, Self> {
        let rest = input.s.strip_prefix(self).ok_or(ParseError)?;
        Ok((self, ParseInput { s: rest, ..input }))
    }
}

impl Parser<'_, Self> for char {
    fn try_parse<'a>(&self, input: ParseInput<'a>) -> ParseResult<'a, Self> {
        let rest = input.s.strip_prefix(*self).ok_or(ParseError)?;
        Ok((*self, ParseInput { s: rest, ..input }))
    }
}

pub const fn nothing(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    Ok(((), input))
}

pub fn newline(input: ParseInput<'_>) -> ParseResult<'_, char> {
    '\n'.try_parse(input)
}

pub fn space(input: ParseInput<'_>) -> ParseResult<'_, char> {
    ' '.or('\t').try_parse(input)
}

pub fn spaces(input: ParseInput<'_>) -> ParseResult<'_, Vec<char>> {
    space.any_amount().try_parse(input)
}

pub fn comment(input: ParseInput<'_>) -> ParseResult<'_, Vec<char>> {
    '#'.before(Predicate(&|c| c != '\n').any_amount())
        .try_parse(input)
}

pub fn trail(input: ParseInput<'_>) -> ParseResult<'_, Vec<Option<Vec<char>>>> {
    spaces
        .before(comment.maybe().followed_by(newline))
        .any_amount()
        .try_parse(input)
}

pub fn whitespace(input: ParseInput<'_>) -> ParseResult<'_, Vec<Option<Vec<char>>>> {
    if input.enclosure_amount > 0 {
        trail.followed_by(spaces).try_parse(input)
    } else {
        spaces.map(|_| Vec::new()).try_parse(input)
    }
}

pub fn eof(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    input.s.is_empty().then_some(((), input)).ok_or(ParseError)
}

pub fn identifier_boundary(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    nothing
        .lookahead(
            Predicate(&|c| !c.is_alphanumeric() && c != '_')
                .map(|_| ())
                .or(eof),
        )
        .try_parse(input)
}

pub fn open_paren(input: ParseInput<'_>) -> ParseResult<'_, char> {
    let (result, input) = '('.try_parse(input)?;
    Ok((
        result,
        ParseInput {
            enclosure_amount: input.enclosure_amount + 1,
            ..input
        },
    ))
}

pub fn close_paren(input: ParseInput<'_>) -> ParseResult<'_, char> {
    let (result, input) = ')'.try_parse(input)?;
    Ok((
        result,
        ParseInput {
            enclosure_amount: input.enclosure_amount - 1,
            ..input
        },
    ))
}

#[derive(Clone, Debug)]
pub enum Value {
    None,
    String(String),
    Number(f64),
    Function,
}

pub fn number(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
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

    Ok((Expression::Literal(Value::Number(result)), input))
}

pub fn identifier_string(input: ParseInput<'_>) -> ParseResult<'_, String> {
    let ((head, tail), input) = Predicate(&|c| c.is_ascii_alphabetic() || c == '_')
        .and(Predicate(&|c| c.is_alphanumeric() || c == '_').any_amount())
        .try_parse(input)?;

    let mut name = vec![head];
    name.extend(tail);

    Ok((name.into_iter().collect(), input))
}

pub fn identifier(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    identifier_string
        .map(Expression::Identifier)
        .try_parse(input)
}

pub fn none(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    "None"
        .followed_by(identifier_boundary)
        .map(|_| Expression::Literal(Value::None))
        .try_parse(input)
}

pub fn string(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
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
    Ok((Expression::Literal(Value::String(s.join(""))), rest))
}

pub fn atom(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    string.or(none).or(number).or(identifier).try_parse(input)
}

#[derive(Debug)]
pub enum UnaryOperation {
    Pos,
    Neg,
    Call,
}

#[derive(Debug)]
pub enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug)]
pub enum Operation {
    Unary(UnaryOperation, Box<Expression>),
    Binary(BinaryOperation, Box<Expression>, Box<Expression>),
}

#[derive(Debug)]
pub enum Expression {
    Literal(Value),
    Identifier(String),
    Operation(Operation),
}

#[derive(Debug)]
pub struct EvaluationError;

impl Operation {
    pub fn evaluate(self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Unary(op, x) => {
                let x = x.evaluate(context)?;
                match (op, x) {
                    (UnaryOperation::Pos, Value::Number(x)) => Ok(Value::Number(x)),
                    (UnaryOperation::Neg, Value::Number(x)) => Ok(Value::Number(-x)),
                    _ => Err(EvaluationError),
                }
            }
            Self::Binary(op, x, y) => {
                let x = x.evaluate(context)?;

                // will need to change for chaining with side effects
                let y = y.evaluate(context)?;

                match (op, x, y) {
                    (BinaryOperation::Add, Value::Number(x), Value::Number(y)) => {
                        Ok(Value::Number(x + y))
                    }
                    (BinaryOperation::Sub, Value::Number(x), Value::Number(y)) => {
                        Ok(Value::Number(x - y))
                    }
                    (BinaryOperation::Mul, Value::Number(x), Value::Number(y)) => {
                        Ok(Value::Number(x * y))
                    }
                    (BinaryOperation::Div, Value::Number(_), Value::Number(0.)) => {
                        Err(EvaluationError)
                    }
                    (BinaryOperation::Div, Value::Number(x), Value::Number(y)) => {
                        Ok(Value::Number(x / y))
                    }
                    _ => Err(EvaluationError),
                }
            }
        }
    }
}

pub type Context = HashMap<String, Value>;

impl Expression {
    pub fn evaluate(self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Literal(v) => Ok(v),
            Self::Identifier(n) => context.get(&n).cloned().ok_or(EvaluationError),
            Self::Operation(op) => op.evaluate(context),
        }
    }
}

pub fn enclosed(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    open_paren
        .before(whitespace)
        .before(expression)
        .followed_by(whitespace)
        .followed_by(close_paren)
        .try_parse(input)
}

pub fn primary(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    enclosed.or(atom).try_parse(input)
}

pub fn args(input: ParseInput<'_>) -> ParseResult<'_, UnaryOperation> {
    open_paren
        .followed_by(whitespace)
        .followed_by(close_paren)
        .map(|_| UnaryOperation::Call)
        .try_parse(input)
}

pub fn call(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let (result, input) = primary.try_parse(input)?;
    let (calls, input) = whitespace.before(args).any_amount().try_parse(input)?;

    let result = calls.into_iter().fold(result, |acc, call| {
        Expression::Operation(Operation::Unary(call, Box::new(acc)))
    });

    Ok((result, input))
}

pub fn expression(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let pos_neg = |input| {
        let pos = '+'.map(|_| UnaryOperation::Pos);
        let neg = '-'.map(|_| UnaryOperation::Neg);

        let (ops, input) = neg
            .or(pos)
            .followed_by(whitespace)
            .any_amount()
            .try_parse(input)?;
        let (result, input) = call.try_parse(input)?;

        let result = ops.into_iter().rev().fold(result, |acc, op| {
            Expression::Operation(Operation::Unary(op, Box::new(acc)))
        });

        Ok((result, input))
    };

    let mul_div = |input| {
        let mul = '*'.map(|_| BinaryOperation::Mul);
        let div = '/'.map(|_| BinaryOperation::Div);

        let ((initial_term, terms), input) = pos_neg
            .and(
                whitespace
                    .before(mul.or(div))
                    .followed_by(whitespace)
                    .and(pos_neg)
                    .any_amount(),
            )
            .try_parse(input)?;

        let result = terms.into_iter().fold(initial_term, |acc, (op, term)| {
            Expression::Operation(Operation::Binary(op, Box::new(acc), Box::new(term)))
        });

        Ok((result, input))
    };

    let add_sub = |input| {
        let add = '+'.map(|_| BinaryOperation::Add);
        let sub = '-'.map(|_| BinaryOperation::Sub);

        let ((initial_term, terms), input) = mul_div
            .and(
                whitespace
                    .before(add.or(sub))
                    .followed_by(whitespace)
                    .and(mul_div)
                    .any_amount(),
            )
            .try_parse(input)?;

        let result = terms.into_iter().fold(initial_term, |acc, (op, term)| {
            Expression::Operation(Operation::Binary(op, Box::new(acc), Box::new(term)))
        });

        Ok((result, input))
    };

    add_sub.try_parse(input)
}

#[derive(Debug)]
pub enum Statement {
    Assignment(String, Expression),
}

impl Statement {
    pub fn execute(self, context: &mut Context) -> Option<EvaluationError> {
        match self {
            Self::Assignment(name, value) => match value.evaluate(context) {
                Ok(value) => {
                    context.insert(name, value);
                    None
                }
                Err(e) => Some(e),
            },
        }
    }
}

pub fn assignable(input: ParseInput<'_>) -> ParseResult<'_, String> {
    let (name, input) = identifier_string.try_parse(input)?;
    match name.as_str() {
        "None" => Err(ParseError),
        _ => Ok((name, input)),
    }
}

pub fn statement(input: ParseInput<'_>) -> ParseResult<'_, Statement> {
    let assignment = assignable
        .followed_by(spaces)
        .followed_by('=')
        .followed_by(spaces)
        .and(expression)
        .map(|(name, value)| Statement::Assignment(name, value));
    assignment.try_parse(input)
}

pub fn block(input: ParseInput<'_>) -> ParseResult<'_, Vec<Statement>> {
    let (indentation, _) = spaces.try_parse(input)?;
    let indentation_length = indentation.into_iter().map(char::len_utf8).sum();
    let indentation = &input.s[..indentation_length];

    // check whether the indentation level increased
    if !input.in_global_scope
        && input
            .indentation
            .strip_prefix(indentation)
            .into_iter()
            .all(str::is_empty)
    {
        Err(ParseError)?;
    }

    let initial_indentation = input.indentation;
    let input = ParseInput {
        indentation,
        ..input
    };

    let (result, input) = indentation
        .before(statement)
        .followed_by(trail)
        .any_amount()
        .try_parse(input)?;

    let input = ParseInput {
        indentation: initial_indentation,
        ..input
    };

    Ok((result, input))
}
