pub mod parser;

use parser::{ParseError, ParseResult, Parser};

fn digit(s: &str) -> ParseResult<'_, u32> {
    let (a, s) = s.split_at_checked(1).ok_or(ParseError)?;
    a.chars()
        .next()
        .and_then(|c| c.to_digit(10))
        .map(|d| (d, s))
        .ok_or(ParseError)
}

fn space(s: &str) -> ParseResult<'_, char> {
    ' '.or(&'\t').try_parse(s)
}

fn spaces(s: &str) -> ParseResult<'_, Vec<char>> {
    space.any_amount().try_parse(s)
}

fn number(s: &str) -> ParseResult<'_, f64> {
    let ((d, ds), mut s) = digit.at_least_one().try_parse(s)?;
    let mut result = ds
        .into_iter()
        .fold(f64::from(d), |acc, d| acc.mul_add(10., d.into()));

    if let Ok(((d, ds), rest)) = '.'.before(&digit.at_least_one()).try_parse(s) {
        s = rest;

        let (fractional_part, _) = ds
            .into_iter()
            .fold((f64::from(d) / 10., 100.), |(acc, m), d| {
                (acc + f64::from(d) / m, m * 10.)
            });
        result += fractional_part;
    }

    Ok((result, s))
}

fn main() {
    println!("{:?}", "pineapple".try_parse("pineapple pizza"));
    println!("{:?}", "pineapple".at_least_one().try_parse(""));
    println!(
        "{:?}",
        "pineapple".at_least_one().try_parse("pineapplepineapple")
    );
    println!(
        "{:?}",
        number
            .followed_by(&spaces)
            .and(&number)
            .try_parse("51.23     1234.1234")
    );
}
