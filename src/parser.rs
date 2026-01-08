#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

#[derive(Debug)]
pub struct ParseError;

#[derive(Clone, Copy, Debug)]
pub struct ParseInput<'a> {
    enclosed: bool,
    s: &'a str,
}

impl<'a> From<&'a str> for ParseInput<'a> {
    fn from(value: &'a str) -> Self {
        ParseInput {
            enclosed: false,
            s: value,
        }
    }
}

pub type ParseResult<'a, T> = Result<(T, ParseInput<'a>), ParseError>;

pub trait Parser<'a, T> {
    fn try_parse(&self, input: ParseInput<'a>) -> ParseResult<'a, T>;

    fn maybe(&self) -> impl Parser<'a, Option<T>> {
        |input| match self.try_parse(input) {
            Ok((x, input)) => Ok((Some(x), input)),
            Err(_) => Ok((None, input)),
        }
    }

    fn any_amount(&self) -> impl Parser<'a, Vec<T>> {
        |mut input| {
            let mut result = Vec::new();
            while let Ok((x, rest)) = self.try_parse(input) {
                result.push(x);
                input = rest;
            }

            Ok((result, input))
        }
    }

    fn at_least_one(&self) -> impl Parser<'a, (T, Vec<T>)> {
        |input| {
            let (x, mut input) = self.try_parse(input)?;

            let mut xs = Vec::new();
            while let Ok((x, rest)) = self.try_parse(input) {
                xs.push(x);
                input = rest;
            }

            Ok(((x, xs), input))
        }
    }

    fn followed_by<U>(&self, other: &impl Parser<'a, U>) -> impl Parser<'a, T> {
        |input| {
            let (x, input) = self.try_parse(input)?;
            let (_, input) = other.try_parse(input)?;

            Ok((x, input))
        }
    }

    fn before<U>(&self, other: &impl Parser<'a, U>) -> impl Parser<'a, U> {
        |input| {
            let (_, input) = self.try_parse(input)?;
            let (x, input) = other.try_parse(input)?;

            Ok((x, input))
        }
    }

    fn and<U>(&self, other: &impl Parser<'a, U>) -> impl Parser<'a, (T, U)> {
        |input| {
            let (x, input) = self.try_parse(input)?;
            let (y, input) = other.try_parse(input)?;

            Ok(((x, y), input))
        }
    }

    fn or(&self, other: &impl Parser<'a, T>) -> impl Parser<'a, T> {
        |input| match self.try_parse(input) {
            r @ Ok(_) => r,
            Err(_) => other.try_parse(input),
        }
    }

    fn lookahead<U>(&self, other: &impl Parser<'a, U>) -> impl Parser<'a, T> {
        |input| {
            let (x, input) = self.try_parse(input)?;
            let _ = other.try_parse(input)?;

            Ok((x, input))
        }
    }

    fn map<U>(&self, f: &impl Fn(T) -> U) -> impl Parser<'a, U> {
        |input| match self.try_parse(input) {
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
    if input.enclosed {
        ' '.or(&'\t').or(&newline).try_parse(input)
    } else {
        ' '.or(&'\t').try_parse(input)
    }
}

pub fn spaces(input: ParseInput<'_>) -> ParseResult<'_, Vec<char>> {
    space.any_amount().try_parse(input)
}

pub fn eof(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    input.s.is_empty().then_some(((), input)).ok_or(ParseError)
}

pub fn identifier_boundary(input: ParseInput<'_>) -> ParseResult<'_, ()> {
    nothing
        .lookahead(
            &Predicate(&|c| !c.is_alphanumeric() && c != '_')
                .map(&|_| ())
                .or(&eof),
        )
        .try_parse(input)
}

#[derive(Debug)]
pub enum Expression {
    None,
    String(String),
    Number(f64),
    Identifier(String),
}

pub fn number(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let digit = Predicate(&|c| c.is_ascii_digit());

    let ((whole, fractional), input) = digit
        .any_amount()
        .and(&'.'.before(&digit.at_least_one()).maybe())
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

    Ok((Expression::Number(result), input))
}

pub fn identifier(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let ((head, tail), input) = Predicate(&|c| c.is_ascii_alphabetic() || c == '_')
        .and(&Predicate(&|c| c.is_alphanumeric() || c == '_').any_amount())
        .try_parse(input)?;

    let mut name = vec![head];
    name.extend(tail);

    Ok((Expression::Identifier(name.into_iter().collect()), input))
}

pub fn none(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    "None"
        .followed_by(&identifier_boundary)
        .map(&|_| Expression::None)
        .try_parse(input)
}

pub fn string(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    fn escaped(input: ParseInput<'_>) -> ParseResult<'_, &str> {
        let (_, rest) = '\\'.and(&Predicate(&|_| true)).try_parse(input)?;
        let taken = &input.s[..input.s.len() - rest.s.len()];

        Ok((taken, rest))
    }

    fn unescaped(input: ParseInput<'_>) -> ParseResult<'_, &str> {
        let (_, rest) = Predicate(&|c| c != '"').try_parse(input)?;
        let taken = &input.s[..input.s.len() - rest.s.len()];

        Ok((taken, rest))
    }

    let (s, rest) = '"'
        .before(&escaped.or(&unescaped).any_amount())
        .followed_by(&'"')
        .try_parse(input)?;
    Ok((Expression::String(s.join("")), rest))
}

pub fn expression(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    string
        .or(&none)
        .or(&number)
        .or(&identifier)
        .try_parse(input)
}
