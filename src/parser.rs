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
