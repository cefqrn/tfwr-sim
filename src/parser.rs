#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::collections::HashMap;

#[derive(Debug)]
pub struct ParseError;

#[derive(Clone, Copy, Debug)]
pub struct ParseInput<'a> {
    enclosure_amount: usize,
    indentation: &'a str,
    s: &'a str,
    fully_consumed: bool,
}

impl<'a> From<&'a str> for ParseInput<'a> {
    fn from(value: &'a str) -> Self {
        ParseInput {
            enclosure_amount: 0,
            indentation: "",
            s: value.trim(),
            fully_consumed: false,
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

    fn ignore(self) -> impl Parser<'a, ()> {
        self.map(|_| ())
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

pub fn end_of_line(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    spaces
        .ignore()
        .followed_by(comment.maybe())
        .followed_by(newline.ignore().or(eof))
        .try_parse(input)
}

pub fn up_to_next_statement(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    end_of_line.at_least_one().ignore().try_parse(input)
}

pub fn whitespace(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    if input.enclosure_amount > 0 {
        end_of_line
            .any_amount()
            .ignore()
            .followed_by(spaces)
            .try_parse(input)
    } else {
        spaces.ignore().try_parse(input)
    }
}

pub fn eof(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    if input.fully_consumed {
        Err(ParseError)?;
    }

    let input = ParseInput {
        fully_consumed: true,
        ..input
    };

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
    Bool(bool),
    Function(Vec<String>, Vec<Statement>),
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

pub fn keyword<T: Copy>(s: &str, parsed_value: T) -> impl Parser<'_, T> {
    s.followed_by(identifier_boundary)
        .map(move |_| parsed_value)
}

pub fn atom(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let none = "None".map(|_| Expression::Literal(Value::None));
    let true_ = "True".map(|_| Expression::Literal(Value::Bool(true)));
    let false_ = "False".map(|_| Expression::Literal(Value::Bool(false)));

    string
        .or(none.or(true_).or(false_).followed_by(identifier_boundary))
        .or(number)
        .or(identifier)
        .try_parse(input)
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOperation {
    Pos,
    Neg,
    Call,
}

#[derive(Clone, Copy, Debug)]
pub enum ArithmeticOperation {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug)]
pub enum BinaryOperation {
    Arithmetic(ArithmeticOperation),
}

#[derive(Clone, Debug)]
pub enum Operation {
    Unary(UnaryOperation, Box<Expression>),
    Binary(BinaryOperation, Box<Expression>, Box<Expression>),
}

#[derive(Clone, Debug)]
pub enum Expression {
    Literal(Value),
    Identifier(String),
    Operation(Operation),
}

#[derive(Debug)]
pub struct EvaluationError;

impl Value {
    pub fn as_num(self) -> Result<Self, EvaluationError> {
        match self {
            Self::Number(_) => Ok(self),
            Self::Bool(true) => Ok(Self::Number(1.)),
            Self::Bool(false) => Ok(Self::Number(0.)),
            _ => Err(EvaluationError),
        }
    }

    pub fn as_bool(self) -> Result<Self, EvaluationError> {
        match self {
            Self::Bool(_) => Ok(self),
            Self::Number(n) => Ok(Self::Bool(n != 0.)),
            _ => Ok(Self::Bool(true)),
        }
    }
}

impl Operation {
    pub fn evaluate(self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Unary(op, x) => {
                let x = x.evaluate(context)?;
                match op {
                    UnaryOperation::Pos => match x {
                        Value::Number(_) | Value::Bool(_) => Ok(x),
                        _ => Err(EvaluationError),
                    },
                    UnaryOperation::Neg => match x.as_num()? {
                        Value::Number(x) => Ok(Value::Number(-x)),
                        _ => Err(EvaluationError),
                    },
                    UnaryOperation::Call => Err(EvaluationError), // TODO
                }
            }
            Self::Binary(op, x, y) => {
                let x = x.evaluate(context)?;

                match op {
                    BinaryOperation::Arithmetic(op) => {
                        let Value::Number(x) = x.as_num()? else {
                            panic!("as_num should return a number")
                        };
                        let Value::Number(y) = y.evaluate(context)?.as_num()? else {
                            panic!("as_num should return a number")
                        };

                        match op {
                            ArithmeticOperation::Add => Ok(Value::Number(x + y)),
                            ArithmeticOperation::Sub => Ok(Value::Number(x - y)),
                            ArithmeticOperation::Mul => Ok(Value::Number(x * y)),
                            ArithmeticOperation::Div if y == 0. => Err(EvaluationError),
                            ArithmeticOperation::Div => Ok(Value::Number(x / y)),
                        }
                    }
                }
            }
        }
    }
}

pub type Variable = Option<Value>;
pub type Context = HashMap<String, (Variable, Vec<Variable>)>;

impl Expression {
    pub fn evaluate(self, context: &mut Context) -> Result<Value, EvaluationError> {
        match self {
            Self::Literal(v) => Ok(v),
            Self::Operation(op) => op.evaluate(context),
            Self::Identifier(n) => match context.get(&n) {
                None => Err(EvaluationError), // variable does not exist
                Some((v, _)) => v.clone().ok_or(EvaluationError),
            },
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
        let mul = '*'.map(|_| ArithmeticOperation::Mul);
        let div = '/'.map(|_| ArithmeticOperation::Div);

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
            Expression::Operation(Operation::Binary(
                BinaryOperation::Arithmetic(op),
                Box::new(acc),
                Box::new(term),
            ))
        });

        Ok((result, input))
    };

    let add_sub = |input| {
        let add = '+'.map(|_| ArithmeticOperation::Add);
        let sub = '-'.map(|_| ArithmeticOperation::Sub);

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
            Expression::Operation(Operation::Binary(
                BinaryOperation::Arithmetic(op),
                Box::new(acc),
                Box::new(term),
            ))
        });

        Ok((result, input))
    };

    add_sub.try_parse(input)
}

#[derive(Clone, Debug)]
pub enum Statement {
    Assignment(String, Expression),
    If(Vec<(Expression, Vec<Statement>)>, Vec<Statement>),
    Def(String, Vec<String>, Vec<Statement>),
}

pub fn declare(context: &mut Context, name: String) {
    context.insert(name, (None, Vec::new()));
}

pub fn declare_locally(context: &mut Context, name: &str) {
    let (top, stack) = context.get_mut(name).expect("already added to context");
    stack.push(top.take());
}

pub fn assign(context: &mut Context, name: &str, value: Value) {
    let (top, _) = context.get_mut(name).expect("already added to context");
    top.replace(value);
}

#[must_use]
pub fn assigned_to_in(statements: &[Statement]) -> Vec<&str> {
    // TODO: globals
    let mut result = Vec::new();
    for s in statements {
        match s {
            Statement::Assignment(name, _) => result.push(name.as_str()),
            Statement::Def(name, _, _) => result.push(name.as_str()),
            Statement::If(if_branches, else_branch) => {
                for (_, body) in if_branches {
                    result.extend(assigned_to_in(body));
                }

                result.extend(assigned_to_in(else_branch));
            }
        }
    }

    result
}

impl Statement {
    pub fn execute(self, context: &mut Context) -> Option<EvaluationError> {
        match self {
            Self::Assignment(name, value) => match value.evaluate(context) {
                Ok(value) => {
                    assign(context, &name, value);
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
                assign(context, &name, value);
                None
            }
        }
    }
}

pub fn assignable(input: ParseInput<'_>) -> ParseResult<'_, String> {
    let (name, input) = identifier_string.try_parse(input)?;
    match name.as_str() {
        "None" | "True" | "False" => Err(ParseError),
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

    let single_line = assignment.followed_by(up_to_next_statement);

    let if_ = "if"
        .before(spaces)
        .before(expression)
        .followed_by(spaces)
        .followed_by(':')
        .followed_by(up_to_next_statement)
        .and(block)
        .and(
            "elif"
                .before(spaces)
                .before(expression)
                .followed_by(spaces)
                .followed_by(':')
                .followed_by(up_to_next_statement)
                .and(block)
                .any_amount(),
        )
        .and(
            "else"
                .before(spaces)
                .before(':')
                .followed_by(up_to_next_statement)
                .before(block)
                .maybe(),
        )
        .map(|((initial, elifs), else_)| {
            let mut possibilities = vec![initial];
            possibilities.extend(elifs);

            Statement::If(possibilities, else_.unwrap_or_else(Vec::new))
        });

    let function = "def"
        .before(spaces)
        .before(assignable)
        .followed_by(spaces)
        .followed_by(open_paren)
        .and(
            whitespace
                .before(assignable)
                .followed_by(whitespace.before(',').maybe())
                .any_amount(),
        )
        .followed_by(whitespace)
        .followed_by(close_paren)
        .followed_by(spaces)
        .followed_by(':')
        .followed_by(up_to_next_statement)
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
    let (indentation, _) = spaces.try_parse(input)?;
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
    for var in assigned_to_in(&result) {
        context.insert(var.to_owned(), (None, Vec::new()));
    }

    Ok(((context, result), input))
}
