use evaluation::Context;
use parsing::Parser;
use tfwr_sim::evaluation;
use tfwr_sim::expression;
use tfwr_sim::parsing;
use tfwr_sim::statement;
use tfwr_sim::value;
use value::Value;

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
        value::parse
            .followed_by(&parsing::spaces)
            .and(&value::parse)
            .followed_by(&parsing::spaces)
            .and(&value::parse)
            .try_parse("51.23     1234    .5".into())
    );
    println!("{:?}", value::parse.try_parse(".".into()));
    println!("{:?}", value::parse.try_parse("5.".into()));

    println!("{:?}", expression::parse.try_parse("pineapple ".into()));
    println!("{:?}", expression::parse.try_parse("1pineapple ".into()));
    println!(
        "{:?}",
        expression::parse
            .try_parse("_ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω123 ".into())
    );
    println!(
        "{:?}",
        expression::parse.try_parse("ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω123 ".into())
    );

    println!("{:?}", expression::parse.try_parse("None".into()));
    println!("{:?}", expression::parse.try_parse("Noneα".into()));
    println!("{:?}", expression::parse.try_parse("None ".into()));

    println!("{:?}", expression::parse.try_parse("\"\" ".into()));
    println!(
        "{:?}",
        expression::parse.try_parse("\"pineapple\npizza\" ".into())
    );
    println!(
        "{:?}",
        expression::parse.try_parse("\"\\\"pineapple\\\" pizza\" ".into())
    );

    println!("{:?}", expression::parse.try_parse("(1)".into()));
    println!(
        "{:?}",
        expression::parse.try_parse(
            "(#comment
        (1 # comment
        )

        ) # comment"
                .into()
        )
    );

    println!("{:?}", expression::parse.try_parse("+  1".into()));
    println!(
        "{:?}",
        expression::parse.try_parse(
            "-
            1"
            .into()
        )
    );
    println!(
        "{:?}",
        expression::parse.try_parse(
            "(+

        1 # comment
        )"
            .into()
        )
    );

    let mut context = Context::new();
    evaluation::declare(&mut context, "pineapple".to_owned());
    evaluation::assign(&mut context, "pineapple", Value::Number(123.));

    let (x, _) = expression::parse.try_parse("-5.6".into()).unwrap();
    println!("{:?}", x.evaluate(&mut context));
    let (x, _) = expression::parse.try_parse("+5.6".into()).unwrap();
    println!("{:?}", x.evaluate(&mut context));

    println!(
        "{:?}",
        expression::parse.try_parse(
            "pineapple (
    ) "
            .into()
        )
    );

    println!(
        "{:?}",
        expression::parse.try_parse(
            "-pineapple (
    ) "
            .into()
        )
    );
    let (x, _) = expression::parse.try_parse("pineapple()".into()).unwrap();
    println!("{:?}", x.evaluate(&mut context));

    println!(
        "{:?}",
        expression::parse.try_parse("--+-++-1 + ---+-+-5".into())
    );
    println!(
        "{:?}",
        expression::parse.try_parse("1 + 2 - 3 * 7 - -5 / 3".into())
    );
    let (x, _) = expression::parse
        .try_parse("1 + 2 - 3 * 7 - -5 / 3".into())
        .unwrap();
    println!("{:?}", x.evaluate(&mut context));

    let (x, _) = expression::parse
        .try_parse("567 * -pineapple".into())
        .unwrap();
    println!("{:?}", x.evaluate(&mut context));

    let (x, _) = statement::statement.try_parse("pizza = 5".into()).unwrap();
    evaluation::declare(&mut context, "pizza".to_owned());
    println!("{context:?} {x:?}");
    x.execute(&mut context);
    println!("{context:?}");

    let ((mut context, x), _) = statement::module
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

    println!("{:?}", statement::statement.try_parse("None = 5".into()));

    let (x, _) = expression::parse.try_parse("+True".into()).unwrap();
    println!("{x:?}");
    println!("{:?}", x.evaluate(&mut context));
    let (x, _) = expression::parse.try_parse("-True".into()).unwrap();
    println!("{x:?}");
    println!("{:?}", x.evaluate(&mut context));

    let ((mut context, x), _) = statement::module
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

    let ((mut context, x), _) = statement::module
        .try_parse(
            "
if True:
    x = 5
elif True:
    x = 6
elif True:
    x = 7
else:
    x = 8

if False:
    y = 5
elif True:
    y = 6
elif True:
    y = 7
else:
    y = 8

if False:
    z = 5
elif False:
    z = 6
elif True:
    z = 7
else:
    z = 8

if False:
    w = 5
elif False:
    w = 6
elif False:
    w = 7
else:
    w = 8
"
            .into(),
        )
        .unwrap();
    println!("{context:?} {x:?}");
    for statement in x {
        statement.execute(&mut context);
    }
    println!("{context:?}");

    println!("{:?}", statement::module.try_parse("x = 1 y = 2".into()));

    let ((mut context, x), _) = statement::module
        .try_parse(
            "
def f(a, b):
    k = a + b

n = 5
"
            .into(),
        )
        .unwrap();

    println!("{context:?} {x:?}");
    for s in x {
        s.execute(&mut context);
    }
    println!("{context:?}");

    let ((mut context, x), _) = statement::module.try_parse("x = x + 1".into()).unwrap();
    evaluation::assign(&mut context, "x", Value::Number(1.));
    println!("{context:?} {x:?}");
    for s in x.clone() {
        s.execute(&mut context);
    }
    println!("{context:?}");
    for s in x {
        s.execute(&mut context);
    }
    println!("{context:?}");

    let ((context, x), _) = statement::module
        .try_parse(
            "
def f():
    global a
    a = 5"
                .into(),
        )
        .unwrap();
    println!("{context:?} {x:?}");

    let ((mut context, x), _) = statement::module
        .try_parse(
            "
def f():
    global a
    global b
    if b - 144:
        c = a
        a = b
        b = c + b
        _ = f()

a = 0
b = 1
_ = f()
    "
            .into(),
        )
        .unwrap();
    println!("{context:?} {x:?}");
    for s in x {
        s.execute(&mut context);
    }
    println!("{:?} {:?}", context.get("a"), context.get("b"));

    let ((mut context, x), _) = statement::module
        .try_parse(
            "
def f(a, b):
    global result
    if b - 144:
        _ = f(b, a+b)
    else:
        result = a+b

_ = f(0, 1)
    "
            .into(),
        )
        .unwrap();
    println!("{context:?} {x:?}");
    for s in x {
        s.execute(&mut context);
    }
    println!("{:?}", context.get("result"));
}
