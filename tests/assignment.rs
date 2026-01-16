use tfwr_sim::parsing::Parser;
use tfwr_sim::statement;
use tfwr_sim::value::Value;

mod nested_capture_modification {
    use super::*;

    #[test]
    fn with_later_local_assignment_without_global_is_local() {
        let ((mut context, x), _) = statement::module
            .try_parse(
                "
x = 5
def f():
    def g():
        global x
        x = 123

    g()
    x = 6

    global y
    y = x

f()
"
                .into(),
            )
            .unwrap();

        println!("{context:?}\n{x:?}");
        assert!(x.evaluate(&mut context).is_ok());
        assert_eq!(Value::Number(5.), context.get("x").unwrap());
        assert_eq!(Value::Number(6.), context.get("y").unwrap());
    }

    #[test]
    fn with_local_assignment_with_global_is_local() {
        let ((mut context, x), _) = statement::module
            .try_parse(
                "
x = 5
def f():
    def g():
        global x
        x = 123

    x = 6
    g()

f()
"
                .into(),
            )
            .unwrap();

        println!("{context:?}\n{x:?}");
        assert!(x.evaluate(&mut context).is_ok());
        assert_eq!(Value::Number(5.), context.get("x").unwrap());
    }

    #[test]
    fn without_local_assignment_without_global_is_not_local() {
        let ((mut context, x), _) = statement::module
            .try_parse(
                "
x = 5
def f():
    global a
    global b

    def g():
        global x
        x = 123
    a = x
    g()
    b = x

f()
"
                .into(),
            )
            .unwrap();

        println!("{context:?}\n{x:?}");
        assert!(x.evaluate(&mut context).is_ok());
        assert_eq!(Value::Number(5.), context.get("a").unwrap());
        assert_eq!(Value::Number(123.), context.get("b").unwrap());
        assert_eq!(Value::Number(123.), context.get("x").unwrap());
    }

    #[test]
    fn without_local_assignment_with_global_is_not_local() {
        let ((mut context, x), _) = statement::module
            .try_parse(
                "
x = 5
def f():
    global x
    global a
    global b

    def g():
        global x
        x = 123
    a = x
    g()
    b = x

f()
"
                .into(),
            )
            .unwrap();

        println!("{context:?}\n{x:?}");
        assert!(x.evaluate(&mut context).is_ok());
        assert_eq!(Value::Number(5.), context.get("a").unwrap());
        assert_eq!(Value::Number(123.), context.get("b").unwrap());
        assert_eq!(Value::Number(123.), context.get("x").unwrap());
    }
}
