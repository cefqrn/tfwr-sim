#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

#[derive(Debug)]
pub struct ParseError;

pub type ParseResult<'a, T> = Result<(T, &'a str), ParseError>;

pub trait Parser<'a, T> {
    fn try_parse(&self, s: &'a str) -> ParseResult<'a, T>;

    fn maybe(&self) -> impl Parser<'a, Option<T>> {
        |s| match self.try_parse(s) {
            Ok((x, s)) => Ok((Some(x), s)),
            Err(_) => Ok((None, s)),
        }
    }

    fn any_amount(&self) -> impl Parser<'a, Vec<T>> {
        |mut s| {
            let mut result = Vec::new();
            while let Ok((x, rest)) = self.try_parse(s) {
                result.push(x);
                s = rest;
            }

            Ok((result, s))
        }
    }

    fn at_least_one(&self) -> impl Parser<'a, (T, Vec<T>)> {
        |s| {
            let (x, mut s) = self.try_parse(s)?;

            let mut xs = Vec::new();
            while let Ok((x, rest)) = self.try_parse(s) {
                xs.push(x);
                s = rest;
            }

            Ok(((x, xs), s))
        }
    }

    fn followed_by<U>(&self, other: &impl Parser<'a, U>) -> impl Parser<'a, T> {
        |s| {
            let (x, s) = self.try_parse(s)?;
            let (_, s) = other.try_parse(s)?;

            Ok((x, s))
        }
    }

    fn before<U>(&self, other: &impl Parser<'a, U>) -> impl Parser<'a, U> {
        |s| {
            let (_, s) = self.try_parse(s)?;
            let (x, s) = other.try_parse(s)?;

            Ok((x, s))
        }
    }

    fn and<U>(&self, other: &impl Parser<'a, U>) -> impl Parser<'a, (T, U)> {
        |s| {
            let (x, s) = self.try_parse(s)?;
            let (y, s) = other.try_parse(s)?;

            Ok(((x, y), s))
        }
    }

    fn or(&self, other: &impl Parser<'a, T>) -> impl Parser<'a, T> {
        |s| match self.try_parse(s) {
            r @ Ok(_) => r,
            Err(_) => other.try_parse(s),
        }
    }
}

impl<'b, T, U: Fn(&'b str) -> ParseResult<'b, T>> Parser<'b, T> for U {
    fn try_parse(&self, s: &'b str) -> ParseResult<'b, T> {
        self(s)
    }
}

// Pattern is nightly
// and Fn(char) -> bool conflicts with above
struct Predicate<'a>(&'a dyn Fn(char) -> bool);
impl<'b> Parser<'b, char> for Predicate<'_> {
    fn try_parse(&self, s: &'b str) -> ParseResult<'b, char> {
        let c = s.chars().next().ok_or(ParseError)?;
        self.0(c)
            .then(|| (c, s.strip_prefix(c).expect("got prefix from string")))
            .ok_or(ParseError)
    }
}

impl Parser<'_, Self> for &str {
    fn try_parse<'a>(&self, s: &'a str) -> ParseResult<'a, Self> {
        s.strip_prefix(*self)
            .map(|rest| (*self, rest))
            .ok_or(ParseError)
    }
}

impl Parser<'_, Self> for char {
    fn try_parse<'a>(&self, s: &'a str) -> ParseResult<'a, Self> {
        s.strip_prefix(*self)
            .map(|rest| (*self, rest))
            .ok_or(ParseError)
    }
}

pub fn space(s: &str) -> ParseResult<'_, char> {
    ' '.or(&'\t').try_parse(s)
}

pub fn spaces(s: &str) -> ParseResult<'_, Vec<char>> {
    space.any_amount().try_parse(s)
}

#[derive(Debug)]
pub enum Expression {
    Number(f64),
    Identifier(String),
}

pub fn number(s: &str) -> ParseResult<'_, Expression> {
    let digit = Predicate(&|c| c.is_ascii_digit());

    let ((whole, fractional), s) = digit
        .any_amount()
        .and(&'.'.before(&digit.at_least_one()).maybe())
        .try_parse(s)?;

    let mut result = whole;
    if let Some((d, ds)) = fractional {
        result.push('.');
        result.push(d);
        result.extend(ds);
    } else if result.is_empty() {
        Err(ParseError)?;
    }
    let result = String::from_iter(result).parse().expect("numbers and dot");

    Ok((Expression::Number(result), s))
}

pub fn identifier(s: &str) -> ParseResult<'_, Expression> {
    let ((head, tail), s) = Predicate(&|c| c.is_ascii_alphabetic() || c == '_')
        .and(&Predicate(&|c| c.is_alphanumeric() || c == '_').any_amount())
        .try_parse(s)?;

    let mut name = vec![head];
    name.extend(tail);

    Ok((Expression::Identifier(name.into_iter().collect()), s))
}
