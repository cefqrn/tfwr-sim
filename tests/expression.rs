use expression::Expression;
use tfwr_sim::expression;
use tfwr_sim::module;
use tfwr_sim::parsing::Parser;
use tfwr_sim::value::Value;

mod identifier {
    use super::*;

    mod accepts {
        use super::*;

        #[test]
        fn ascii_letters() {
            if let Ok((Expression::Identifier(name), _)) = Expression::parse.try_parse("Pineapple")
            {
                assert_eq!(name, "Pineapple");
            } else {
                panic!();
            }
        }

        #[test]
        fn underscore() {
            if let Ok((Expression::Identifier(name), _)) = Expression::parse.try_parse("_") {
                assert_eq!(name, "_");
            } else {
                panic!();
            }
        }

        #[test]
        fn numbers_after_first_letter() {
            if let Ok((Expression::Identifier(name), _)) =
                Expression::parse.try_parse("p1i2n3e4a5p6p7l8e9")
            {
                assert_eq!(name, "p1i2n3e4a5p6p7l8e9");
            } else {
                panic!();
            }
        }

        #[test]
        fn non_ascii_alphabetic_after_first_letter() {
            if let Ok((Expression::Identifier(name), _)) =
                Expression::parse.try_parse("aΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω")
            {
                assert_eq!(name, "aΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω");
            } else {
                panic!();
            }
        }

        #[test]
        fn containing_keyword() {
            if let Ok((Expression::Identifier(name), _)) = Expression::parse.try_parse("Nonea") {
                assert_eq!(name, "Nonea");
            }
        }
    }

    mod refuses {
        use super::*;

        #[test]
        fn number_on_first_character() {
            if let Ok((Expression::Identifier(_), _)) = Expression::parse.try_parse("1pineapple") {
                panic!();
            }
        }

        #[test]
        fn non_ascii_on_first_character() {
            if let Ok((Expression::Identifier(_), _)) = Expression::parse.try_parse("Αpineapple") {
                panic!();
            }
        }

        #[test]
        fn keyword() {
            if let Ok((Expression::Identifier(_), _)) = Expression::parse.try_parse("None") {
                panic!();
            }
        }
    }
}

mod literal {
    use super::*;
    use tfwr_sim::value::Value;

    mod keyword {
        use super::*;

        #[test]
        fn none() {
            if let Ok((Expression::Literal(Value::None), _)) = Expression::parse.try_parse("None") {
            } else {
                panic!();
            }
        }
    }

    mod string_literal {
        use super::*;

        #[test]
        fn empty_string() {
            if let Ok((Expression::Literal(Value::String(s)), _)) =
                Expression::parse.try_parse("\"\"")
            {
                assert_eq!(s, "");
            } else {
                panic!();
            }
        }

        #[test]
        fn escaped_newline() {
            if let Ok((Expression::Literal(Value::String(s)), _)) =
                Expression::parse.try_parse("\"pineapple\npizza\"")
            {
                assert_eq!(s, "pineapple\npizza");
            } else {
                panic!();
            }
        }

        #[test]
        fn escaped_quotes() {
            if let Ok((Expression::Literal(Value::String(s)), _)) =
                Expression::parse.try_parse("\"\\\"pineapple\\\" pizza\"")
            {
                assert_eq!(s, "\\\"pineapple\\\" pizza"); // \"pineapple\" pizza
            } else {
                panic!();
            }
        }
    }
}

mod expression_ {
    use super::*;

    mod is_equivalent_when {
        use super::*;

        #[test]
        fn surrounded_by_parens() {
            assert_eq!(
                Expression::parse.try_parse("1"),
                Expression::parse.try_parse("(1)")
            );
        }

        #[test]
        fn surrounded_by_nested_parens() {
            assert_eq!(
                Expression::parse.try_parse("1"),
                Expression::parse.try_parse("((1))")
            );
        }

        #[test]
        fn surrounded_by_parens_and_whitespace_without_newlines() {
            assert_eq!(
                Expression::parse.try_parse("1"),
                Expression::parse.try_parse("( 1 )")
            );
        }

        #[test]
        fn surrounded_by_parens_and_whitespace_with_newlines() {
            assert_eq!(
                Expression::parse.try_parse("1"),
                Expression::parse.try_parse(
                    "(
            1
            )"
                )
            );
        }

        #[test]
        fn surrounded_by_parens_and_whitespace_with_empty_lines() {
            assert_eq!(
                Expression::parse.try_parse("1"),
                Expression::parse.try_parse(
                    "(

            1
            )"
                )
            );
        }

        #[test]
        fn surrounded_by_parens_and_whitespace_and_comments() {
            assert_eq!(
                Expression::parse.try_parse("1"),
                Expression::parse.try_parse(
                    "(#comment
        (1 # comment
        ) # comment

        )"
                )
            );
        }

        #[test]
        fn spaces_between_operation_and_operand() {
            assert_eq!(
                Expression::parse.try_parse("-1"),
                Expression::parse.try_parse("- 1")
            );
        }

        #[test]
        fn newlines_between_operation_and_operand_when_enclosed() {
            assert_eq!(
                Expression::parse.try_parse("-1"),
                Expression::parse.try_parse(
                    "(-
            1)"
                )
            );
        }
    }

    mod correctly_evaluates {
        use super::*;

        #[test]
        fn less_than() {
            let ((mut context, x), _) = module::parse
                .try_parse(
                    "
a = 1 < 2
b = 1 < 1
c = 2 < 1
",
                )
                .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(Value::Bool(true), context.get("a").unwrap());
            assert_eq!(Value::Bool(false), context.get("b").unwrap());
            assert_eq!(Value::Bool(false), context.get("c").unwrap());
        }

        #[test]
        fn equal_to() {
            let ((mut context, x), _) = module::parse
                .try_parse(
                    "
def fun():
    _ = 1

def nuf():
    _ = 1

a = True == True
b = False == False
c = True == False
d = True == 1
e = False == 0
f = \"pineapple\" == \"pineapple\"
g = \"pineapple\" == \"pizza\"
h = fun == fun
i = 1 == fun
j = 1 == 1
k = fun == nuf

inf = 9**9**9**9
nan = inf - inf

l = inf == inf
m = nan == nan
",
                )
                .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Bool(true), context.get("a").unwrap());
            assert_eq!(Value::Bool(true), context.get("b").unwrap());
            assert_eq!(Value::Bool(false), context.get("c").unwrap());
            assert_eq!(Value::Bool(true), context.get("d").unwrap());
            assert_eq!(Value::Bool(true), context.get("e").unwrap());
            assert_eq!(Value::Bool(true), context.get("f").unwrap());
            assert_eq!(Value::Bool(false), context.get("g").unwrap());
            assert_eq!(Value::Bool(true), context.get("h").unwrap());
            assert_eq!(Value::Bool(false), context.get("i").unwrap());
            assert_eq!(Value::Bool(true), context.get("j").unwrap());
            assert_eq!(Value::Bool(false), context.get("k").unwrap());
            assert_eq!(Value::Bool(true), context.get("l").unwrap());
            assert_eq!(Value::Bool(false), context.get("m").unwrap());
        }

        #[test]
        fn not_equal_to() {
            let ((mut context, x), _) = module::parse
                .try_parse(
                    "
def fun():
    _ = 1

def nuf():
    _ = 1

a = True != True
b = False != False
c = True != False
d = True != 1
e = False != 0
f = \"pineapple\" != \"pineapple\"
g = \"pineapple\" != \"pizza\"
h = fun != fun
i = 1 != fun
j = 1 != 1
k = fun != nuf

inf = 9**9**9**9
nan = inf - inf

l = inf != inf
m = nan != nan
",
                )
                .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Bool(false), context.get("a").unwrap());
            assert_eq!(Value::Bool(false), context.get("b").unwrap());
            assert_eq!(Value::Bool(true), context.get("c").unwrap());
            assert_eq!(Value::Bool(false), context.get("d").unwrap());
            assert_eq!(Value::Bool(false), context.get("e").unwrap());
            assert_eq!(Value::Bool(false), context.get("f").unwrap());
            assert_eq!(Value::Bool(true), context.get("g").unwrap());
            assert_eq!(Value::Bool(false), context.get("h").unwrap());
            assert_eq!(Value::Bool(true), context.get("i").unwrap());
            assert_eq!(Value::Bool(false), context.get("j").unwrap());
            assert_eq!(Value::Bool(true), context.get("k").unwrap());
            assert_eq!(Value::Bool(false), context.get("l").unwrap());
            assert_eq!(Value::Bool(true), context.get("m").unwrap());
        }

        #[test]
        fn modulo() {
            let ((mut context, x), _) = module::parse.try_parse("a = 11 % 3").unwrap();
            println!("{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(2.), context.get("a").unwrap());

            let ((mut context, x), _) = module::parse.try_parse("a = -11 % 3").unwrap();
            println!("{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(1.), context.get("a").unwrap());
        }

        #[test]
        fn floor_division() {
            let ((mut context, x), _) = module::parse.try_parse("a = 11 // 3").unwrap();
            println!("{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(3.), context.get("a").unwrap());

            let ((mut context, x), _) = module::parse.try_parse("a = -11 // 3").unwrap();
            println!("{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(-4.), context.get("a").unwrap());
        }

        #[test]
        fn chained_powers() {
            // TODO: make this a power of 2 or check for difference
            let ((mut context, x), _) = module::parse.try_parse("a = -2**-3**-4").unwrap();
            println!("{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(
                Value::Number(-0.991_479_137_495_678_1),
                context.get("a").unwrap()
            );
        }

        #[test]
        fn tuples_in_args() {
            let ((mut context, x), _) = module::parse
                .try_parse(
                    "
def f(a):
    _ = None

def g(a, b):
    _ = None

_ = f(1)
_ = f(1,)
_ = f((1,))
_ = f((1,),)
_ = f((1, 2))
_ = f((1, 2),)
_ = g(1, 2)
_ = g(1, 2,)
_ = g(1, (2,))
_ = g(1, (2,),)
_ = g((1,), 2)
_ = g((1,), 2,)
_ = g((1,), (2,))
_ = g((1,), (2,),)
",
                )
                .unwrap();

            println!("{context:?}\n{x:?}");
            assert!((x.evaluate(&mut context).is_ok()));
        }

        #[test]
        fn pos_bool() {
            // python python converts to int, but tfwr python bools stay bools
            let ((mut context, x), _) = module::parse
                .try_parse(
                    "
a = +True
b = +False
",
                )
                .unwrap();
            println!("{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Bool(true), context.get("a").unwrap());
            assert_eq!(Value::Bool(false), context.get("b").unwrap());
        }

        #[test]
        fn neg_bool() {
            let ((mut context, x), _) = module::parse
                .try_parse(
                    "
a = -True
b = -False
",
                )
                .unwrap();
            println!("{x:?}");
            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(-1.), context.get("a").unwrap());
            assert_eq!(Value::Number(-0.), context.get("b").unwrap());
        }

        #[test]
        fn pos_number_with_whole() {
            let ((mut context, x), _) = module::parse.try_parse("k = +5.75").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(5.75), context.get("k").unwrap());
        }

        #[test]
        fn pos_number_without_whole() {
            let ((mut context, x), _) = module::parse.try_parse("k = +.75").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(0.75), context.get("k").unwrap());
        }

        #[test]
        fn neg_number_with_whole() {
            let ((mut context, x), _) = module::parse.try_parse("k = -5.75").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(-5.75), context.get("k").unwrap());
        }

        #[test]
        fn neg_number_without_whole() {
            let ((mut context, x), _) = module::parse.try_parse("k = -.75").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
            assert_eq!(Value::Number(-0.75), context.get("k").unwrap());
        }

        #[test]
        fn function_call() {
            let ((mut context, x), _) = module::parse
                .try_parse(
                    "
def f():
    _ = 5

_ = f()
",
                )
                .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());
        }

        #[test]
        fn non_function_call() {
            let Ok(((mut context, x), _)) = module::parse.try_parse(
                "
x = 2
_ = x()
",
            ) else {
                println!("failed to parse");
                return;
            };

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_err());
        }

        #[test]
        fn pos_neg_add_sub_mul_div_precedence() {
            let ((mut context, x), _) = module::parse
                .try_parse("k = 1 + +2 - 3 * 7 - -5 / 2")
                .unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(Value::Number(-15.5), context.get("k").unwrap());
        }

        #[test]
        fn pos_neg() {
            let ((mut context, x), _) = module::parse.try_parse("x = --+-1 - ++-+5").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(Value::Number(4.), context.get("x").unwrap());
        }
    }

    mod accepts {
        use super::*;

        #[test]
        fn call() {
            assert!(Expression::parse.try_parse("pineapple()").is_ok());
        }
    }

    mod refuses {
        use super::*;

        #[test]
        fn newlines_between_operation_and_operand_when_not_enclosed() {
            assert_eq!(
                Expression::parse.try_parse("-1"),
                Expression::parse.try_parse(
                    "(-

            1)"
                )
            );
        }

        #[test]
        fn comparison_between_incompatible_types() {
            let ((mut context, x), _) = module::parse.try_parse("a = 5 < \"\"").unwrap();
            println!("{x:?}");
            assert!(x.evaluate(&mut context).is_err());
        }
    }
}

mod tuple {
    use super::*;

    mod accepts {
        use super::*;

        #[test]
        fn unit() {
            let ((mut context, x), _) = module::parse.try_parse("x = ()").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(Value::Tuple(Vec::new()), context.get("x").unwrap());
        }

        #[test]
        fn unenclosed_1_tuple_with_trailing_comma() {
            let ((mut context, x), _) = module::parse.try_parse("x = 1,").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(
                Value::Tuple(vec![Value::Number(1.)]),
                context.get("x").unwrap()
            );
        }

        #[test]
        fn enclosed_1_tuple_with_trailing_comma() {
            let ((mut context, x), _) = module::parse.try_parse("x = (1,)").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(
                Value::Tuple(vec![Value::Number(1.)]),
                context.get("x").unwrap()
            );
        }

        #[test]
        fn unenclosed_2_tuple_without_trailing_comma() {
            let ((mut context, x), _) = module::parse.try_parse("x = 1, 2").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(
                Value::Tuple(vec![Value::Number(1.), Value::Number(2.)]),
                context.get("x").unwrap()
            );
        }

        #[test]
        fn unenclosed_2_tuple_with_trailing_comma() {
            let ((mut context, x), _) = module::parse.try_parse("x = 1, 2,").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(
                Value::Tuple(vec![Value::Number(1.), Value::Number(2.)]),
                context.get("x").unwrap()
            );
        }

        #[test]
        fn enclosed_2_tuple_without_trailing_comma() {
            let ((mut context, x), _) = module::parse.try_parse("x = (1, 2)").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(
                Value::Tuple(vec![Value::Number(1.), Value::Number(2.)]),
                context.get("x").unwrap()
            );
        }

        #[test]
        fn enclosed_2_tuple_with_trailing_comma() {
            let ((mut context, x), _) = module::parse.try_parse("x = (1, 2,)").unwrap();

            println!("{context:?}\n{x:?}");

            assert!(x.evaluate(&mut context).is_ok());

            println!("{context:?}");

            assert_eq!(
                Value::Tuple(vec![Value::Number(1.), Value::Number(2.)]),
                context.get("x").unwrap()
            );
        }
    }

    mod refuses {
        use super::*;

        #[test]
        fn unenclosed_1_tuple_without_trailing_comma() {
            let Ok(((mut context, x), _)) = module::parse.try_parse("x = 1") else {
                println!("failed to parse");
                return;
            };

            println!("{context:?}\n{x:?}");

            let Ok(_) = x.evaluate(&mut context) else {
                println!("failed to evaluate");
                return;
            };

            assert_ne!(
                Value::Tuple(vec![Value::Number(1.)]),
                context.get("x").unwrap()
            );
        }

        #[test]
        fn enclosed_1_tuple_without_trailing_comma() {
            let Ok(((mut context, x), _)) = module::parse.try_parse("x = (1)") else {
                println!("failed to parse");
                return;
            };

            println!("{context:?}\n{x:?}");

            let Ok(_) = x.evaluate(&mut context) else {
                println!("failed to evaluate");
                return;
            };

            assert_ne!(
                Value::Tuple(vec![Value::Number(1.)]),
                context.get("x").unwrap()
            );
        }
    }
}
