use tfwr_sim::parsing::Parser;
use tfwr_sim::statement;
use tfwr_sim::value::Value;

mod statement_ {
    use super::*;

    mod accepts {
        use super::*;

        #[test]
        fn empty_lines_and_comments() {
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

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(
                Some(Value::Number(5.)),
                context.get("a").map(|x| x.take().unwrap())
            );
            assert_eq!(
                Some(Value::Number(6.)),
                context.get("b").map(|x| x.take().unwrap())
            );
            assert_eq!(
                Some(Value::Number(7.)),
                context.get("c").map(|x| x.take().unwrap())
            );
            assert_eq!(
                Some(Value::Number(-23.)),
                context.get("d").map(|x| x.take().unwrap())
            );
        }

        #[test]
        fn nested_ifs() {
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

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            assert_eq!(
                Some(Value::Number(123.)),
                context.get("asdf").map(|x| x.take().unwrap())
            );
            assert_eq!(
                Some(Value::Number(567.)),
                context.get("qwerty").map(|x| x.take().unwrap())
            );
            assert_eq!(None, context.get("ytrewq").unwrap().take());
            assert_eq!(None, context.get("fdsa").unwrap().take());
            assert_eq!(None, context.get("zxcvbn").unwrap().take());
            assert_eq!(None, context.get("nbvcxz").unwrap().take());
        }

        #[test]
        fn elifs_and_else() {
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

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            assert_eq!(Value::Number(5.), context.get("x").unwrap().take().unwrap());
            assert_eq!(Value::Number(6.), context.get("y").unwrap().take().unwrap());
            assert_eq!(Value::Number(7.), context.get("z").unwrap().take().unwrap());
            assert_eq!(Value::Number(8.), context.get("w").unwrap().take().unwrap());
        }

        #[test]
        fn def_without_args() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
def f():
    _ = 5

n = 5
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            println!("{context:?}");
        }

        #[test]
        fn def_with_args() {
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

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            println!("{context:?}");
        }

        #[test]
        fn assignment() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
x = 5
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(5.), context.get("x").unwrap().take().unwrap());
        }

        #[test]
        fn reassignment() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
x = 5
x = 6
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(6.), context.get("x").unwrap().take().unwrap());
        }

        #[test]
        fn reassignment_by_def() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
x = 5
def x():
    _ = 5
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_ne!(Value::Number(5.), context.get("x").unwrap().take().unwrap());
        }

        #[test]
        fn reassignment_with_expression_including_the_assigned_var() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
x = 5
x = x + 1
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(6.), context.get("x").unwrap().take().unwrap());
        }

        #[test]
        fn global_without_call() {
            let ((context, x), _) = statement::module
                .try_parse(
                    "
def f():
    global a
    a = 5
"
                    .into(),
                )
                .unwrap();
            println!("{context:?}\n{x:?}");
        }

        #[test]
        fn global_with_call() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
def f():
    global a
    a = 5

_ = f()
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(5.), context.get("a").unwrap().take().unwrap());
        }

        #[test]
        fn global_with_recursive_call_without_parameters() {
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

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(
                Value::Number(89.),
                context.get("a").unwrap().take().unwrap()
            );
            assert_eq!(
                Value::Number(144.),
                context.get("b").unwrap().take().unwrap()
            );
        }

        #[test]
        fn global_with_recursive_call_with_parameters() {
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

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(
                Value::Number(233.),
                context.get("result").unwrap().take().unwrap()
            );
        }

        #[test]
        fn global_in_non_definition_nested_block() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
def f():
    if False:
        global x
    x = 6

x = 5
_ = f()
# quick_print(x)  # 6
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(Value::Number(6.), context.get("x").unwrap().take().unwrap());
        }

        #[test]
        fn or_short_circuits() {
            let ((mut context, x), _) = statement::module(
                "
def f(x):
    global side_effect
    side_effect = x

k = 5 or f(6)
"
                .into(),
            )
            .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(Value::Number(5.), context.get("k").unwrap().take().unwrap());
            assert_eq!(None, context.get("side_effect").unwrap().take());
        }

        #[test]
        fn and_short_circuits() {
            let ((mut context, x), _) = statement::module(
                "
def f(x):
    global side_effect
    side_effect = x

k = 0 and f(6)
"
                .into(),
            )
            .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(Value::Number(0.), context.get("k").unwrap().take().unwrap());
            assert_eq!(None, context.get("side_effect").unwrap().take());
        }

        #[test]
        fn while_loop() {
            let ((mut context, x), _) = statement::module(
                "
a = 0
b = 1
while b < 1000:
    c = a + b
    a = b
    b = c
"
                .into(),
            )
            .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(
                Value::Number(987.),
                context.get("a").unwrap().take().unwrap()
            );
            assert_eq!(
                Value::Number(1597.),
                context.get("b").unwrap().take().unwrap()
            );
        }

        #[test]
        fn return_with_parameters_and_recursion() {
            let ((mut context, x), _) = statement::module(
                "
def f(a, b):
    if b >= 1000:
        return b
    else:
        return f(b, a+b)

result = f(0, 1)
"
                .into(),
            )
            .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(
                Value::Number(1597.),
                context.get("result").unwrap().take().unwrap()
            );
        }

        #[test]
        fn unit_assignment() {
            let ((mut context, x), _) = statement::module.try_parse("() = ()".into()).unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
        }

        #[test]
        fn unenclosed_1_tuple_assignment() {
            let ((mut context, x), _) = statement::module.try_parse("x, = 1,".into()).unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(1.), context.get("x").unwrap().take().unwrap());
        }

        #[test]
        fn unenclosed_2_tuple_assignment() {
            let ((mut context, x), _) = statement::module.try_parse("x, y = 1, 2".into()).unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(1.), context.get("x").unwrap().take().unwrap());
            assert_eq!(Value::Number(2.), context.get("y").unwrap().take().unwrap());
        }

        #[test]
        fn enclosed_1_tuple_assignment() {
            let ((mut context, x), _) = statement::module.try_parse("(x,) = 1,".into()).unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(1.), context.get("x").unwrap().take().unwrap());
        }

        #[test]
        fn enclosed_2_tuple_assignment() {
            let ((mut context, x), _) =
                statement::module.try_parse("(x, y) = 1, 2".into()).unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(1.), context.get("x").unwrap().take().unwrap());
            assert_eq!(Value::Number(2.), context.get("y").unwrap().take().unwrap());
        }

        #[test]
        fn tuple_stack() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
x = 1, (2, (3, (4, None)))
while x:
    curr, x = x
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(
                Value::Number(4.),
                context.get("curr").unwrap().take().unwrap()
            );
        }

        #[test]
        fn nested_destructuring() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
z = (3, 4), (5, 6)
(x, y), p = z
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(3.), context.get("x").unwrap().take().unwrap());
            assert_eq!(Value::Number(4.), context.get("y").unwrap().take().unwrap());
            assert_eq!(
                Value::Tuple(vec![Value::Number(5.), Value::Number(6.)]),
                context.get("p").unwrap().take().unwrap()
            );
        }

        #[test]
        fn for_loop() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
def f():
    global x
    x = 1

def g():
    global y
    y = 2

def h():
    global z
    z = 3

for fn in f, g, h:
    _ = fn()
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(1.), context.get("x").unwrap().take().unwrap());
            assert_eq!(Value::Number(2.), context.get("y").unwrap().take().unwrap());
            assert_eq!(Value::Number(3.), context.get("z").unwrap().take().unwrap());
        }

        #[test]
        fn bare_call() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
def f():
    global k
    k = 5

f()
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(5.), context.get("k").unwrap().take().unwrap());
        }

        #[test]
        fn returning_functions() {
            let ((mut context, x), _) = statement::module
                .try_parse(
                    "
a = 0
def f():
    global x
    global y
    global z
    global a
    a = a + 1
    x = a
    def g():
        global y
        global z
        global a
        a = a + 1
        y = a
        def h():
            global z
            global a
            a = a + 1
            z = a

        return h

    return g

f()()()
"
                    .into(),
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(3.), context.get("a").unwrap().take().unwrap());
            assert_eq!(Value::Number(1.), context.get("x").unwrap().take().unwrap());
            assert_eq!(Value::Number(2.), context.get("y").unwrap().take().unwrap());
            assert_eq!(Value::Number(3.), context.get("z").unwrap().take().unwrap());
        }
    }

    mod refuses {
        use super::*;

        #[test]
        fn multiple_statements_on_the_same_line() {
            assert!(statement::module.try_parse("x = 1 y = 2".into()).is_err());
        }

        #[test]
        fn assigning_to_keyword() {
            assert!(statement::module.try_parse("None = 5".into()).is_err());
        }
    }
}
