pub mod parser;

use parser::Parser;

fn main() {
    println!("{:?}", "pineapple".try_parse("pineapple pizza".into()));
    println!("{:?}", "pineapple".at_least_one().try_parse("".into()));
    println!(
        "{:?}",
        "pineapple"
            .at_least_one()
            .try_parse("pineapplepineapple".into())
    );

    println!(
        "{:?}",
        parser::number
            .followed_by(&parser::spaces)
            .and(&parser::number)
            .followed_by(&parser::spaces)
            .and(&parser::number)
            .try_parse("51.23     1234    .5".into())
    );
    println!("{:?}", parser::number.try_parse(".".into()));
    println!("{:?}", parser::number.try_parse("5.".into()));

    println!("{:?}", parser::identifier.try_parse("pineapple ".into()));
    println!("{:?}", parser::identifier.try_parse("1pineapple ".into()));
    println!(
        "{:?}",
        parser::identifier
            .try_parse("_ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω123 ".into())
    );
    println!(
        "{:?}",
        parser::identifier
            .try_parse("ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω123 ".into())
    );

    println!("{:?}", parser::expression.try_parse("None".into()));
    println!("{:?}", parser::expression.try_parse("Noneα".into()));
    println!("{:?}", parser::expression.try_parse("None ".into()));

    println!("{:?}", parser::expression.try_parse("\"\" ".into()));
    println!(
        "{:?}",
        parser::expression.try_parse("\"pineapple\npizza\" ".into())
    );
    println!(
        "{:?}",
        parser::expression.try_parse("\"\\\"pineapple\\\" pizza\" ".into())
    );

    println!("{:?}", parser::expression.try_parse("(1)".into()));
    println!(
        "{:?}",
        parser::expression.try_parse(
            "(#comment
        (1 # comment
        )

        ) # comment"
                .into()
        )
    );

    println!("{:?}", parser::expression.try_parse("+  1".into()));
    println!(
        "{:?}",
        parser::expression.try_parse(
            "-
            1"
            .into()
        )
    );
    println!(
        "{:?}",
        parser::expression.try_parse(
            "(+

        1 # comment
        )"
            .into()
        )
    );

    let (x, _) = parser::expression.try_parse("-5.6".into()).unwrap();
    println!("{:?}", x.evaluate());
    let (x, _) = parser::expression.try_parse("+5.6".into()).unwrap();
    println!("{:?}", x.evaluate());

    println!(
        "{:?}",
        parser::expression.try_parse(
            "pineapple (
    ) "
            .into()
        )
    );

    println!(
        "{:?}",
        parser::expression.try_parse(
            "-pineapple (
    ) "
            .into()
        )
    );
    let (x, _) = parser::expression.try_parse("pineapple()".into()).unwrap();
    println!("{:?}", x.evaluate());
}
