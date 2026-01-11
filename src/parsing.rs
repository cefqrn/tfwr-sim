use crate::value;

#[derive(Debug)]
pub struct ParseError;

#[derive(Clone, Copy, Debug)]
pub struct ParseInput<'a> {
    pub enclosure_amount: usize,
    pub indentation: &'a str,
    pub s: &'a str,
    pub fully_consumed: bool,
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

    fn map_to<U: Copy>(self, x: U) -> impl Parser<'a, U> {
        self.map(move |_| x)
    }

    fn ignore(self) -> impl Parser<'a, ()> {
        self.map_to(())
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
pub struct Predicate<'a>(pub &'a dyn Fn(char) -> bool);
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

pub fn identifier_string(input: ParseInput<'_>) -> ParseResult<'_, String> {
    let ((head, tail), input) = Predicate(&|c| c.is_ascii_alphabetic() || c == '_')
        .and(Predicate(&|c| c.is_alphanumeric() || c == '_').any_amount())
        .try_parse(input)?;

    let mut name = vec![head];
    name.extend(tail);

    Ok((name.into_iter().collect(), input))
}

// TODO: better way to declare keywords
pub fn assignable(input: ParseInput<'_>) -> ParseResult<'_, String> {
    let (name, input) = identifier_string.try_parse(input)?;

    if name == "global" {
        Err(ParseError)?;
    }

    // can't assign to a keyword
    value::parse(name.as_str().into())
        .is_err()
        .then_some((name, input))
        .ok_or(ParseError)
}
