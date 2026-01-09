pub mod parser;

use parser::{Context, Parser};

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

    let mut context = Context::new();
    context.insert("pineapple".to_owned(), parser::Value::Number(123.));

    let (x, _) = parser::expression.try_parse("-5.6".into()).unwrap();
    println!("{:?}", x.evaluate(&mut context));
    let (x, _) = parser::expression.try_parse("+5.6".into()).unwrap();
    println!("{:?}", x.evaluate(&mut context));

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
    println!("{:?}", x.evaluate(&mut context));

    println!(
        "{:?}",
        parser::expression.try_parse("--+-++-1 + ---+-+-5".into())
    );
    println!(
        "{:?}",
        parser::expression.try_parse("1 + 2 - 3 * 7 - -5 / 3".into())
    );
    let (x, _) = parser::expression
        .try_parse("1 + 2 - 3 * 7 - -5 / 3".into())
        .unwrap();
    println!("{:?}", x.evaluate(&mut context));

    let (x, _) = parser::expression
        .try_parse("567 * -pineapple".into())
        .unwrap();
    println!("{:?}", x.evaluate(&mut context));

    let (x, _) = parser::statement.try_parse("pizza = 5".into()).unwrap();
    println!("{context:?} {x:?}");
    x.execute(&mut context);
    println!("{context:?}");

    let (x, _) = parser::block
        .try_parse(
            "
a = 5
b = 6  # pineapple

c = 7

# pizza
d = a * -b + c
"
            .into(),
        )
        .unwrap();
    println!("{context:?} {x:?}");
    for statement in x {
        statement.execute(&mut context);
    }
    println!("{context:?}");

    println!("{:?}", parser::statement.try_parse("None = 5".into()));

    let (x, _) = parser::expression.try_parse("+True".into()).unwrap();
    println!("{x:?}");
    println!("{:?}", x.evaluate(&mut context));
    let (x, _) = parser::expression.try_parse("-True".into()).unwrap();
    println!("{x:?}");
    println!("{:?}", x.evaluate(&mut context));

    let mut context = Context::new();

    let (x, _) = parser::block
        .try_parse(
            "
if 999:

    asdf = 123
    if 777:

        qwerty = 567  # pizza

    if 0:
        ytrewq = 765

if 0:
    fdsa = 321
    # pineapple
    if 555:
        zxcvbn = 789
    if 0:
        nbvcxz = 987
"
            .into(),
        )
        .unwrap();
    println!("{context:?} {x:?}");
    for statement in x {
        statement.execute(&mut context);
    }
    println!("{context:?}");
}
