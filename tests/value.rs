use tfwr_sim::parsing::Parser;
use tfwr_sim::value;
use value::Value;

mod number {
    use super::*;

    mod accepts {
        use super::*;

        #[test]
        fn zero() {
            match value::parse.try_parse("0".into()) {
                Ok((Value::Number(0.), _)) => {}
                _ => panic!(),
            }
        }

        #[test]
        fn nonzero_whole() {
            match value::parse.try_parse("1234".into()) {
                Ok((Value::Number(1234.), _)) => {}
                _ => panic!(),
            }
        }

        #[test]
        fn zero_with_fractional() {
            match value::parse.try_parse("0.25".into()) {
                Ok((Value::Number(0.25), _)) => {}
                _ => panic!(),
            }
        }

        #[test]
        fn nonzero_with_fractional() {
            match value::parse.try_parse("12.25".into()) {
                Ok((Value::Number(12.25), _)) => {}
                _ => panic!(),
            }
        }

        #[test]
        fn fractional_without_whole() {
            match value::parse.try_parse(".25".into()) {
                Ok((Value::Number(0.25), _)) => {}
                _ => panic!(),
            }
        }
    }

    mod refuses {
        use super::*;

        #[test]
        fn single_dot() {
            if let Ok((Value::Number(_), _)) = value::parse.try_parse(".".into()) {
                panic!()
            }
        }

        #[test]
        fn whole_with_dot() {
            // doesn't include the dot in the number
            match value::parse.followed_by('.').try_parse("5.".into()) {
                Ok((Value::Number(5.), _)) => {}
                _ => panic!(),
            }
        }
    }
}
