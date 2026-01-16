use tfwr_sim::parsing::Parser;

mod string_parser {
    use super::*;

    mod accepts {
        use super::*;

        #[test]
        fn itself() {
            assert!("pineapple".try_parse("pineapple").is_ok());
        }

        #[test]
        fn a_string_starting_with_it() {
            assert!("pineapple".try_parse("pineapple pizza").is_ok());
        }

        #[test]
        fn an_empty_string_if_the_parser_is_empty() {
            assert!("".try_parse("").is_ok());
        }
    }

    mod refuses {
        use super::*;

        #[test]
        fn an_empty_string_if_the_parser_is_nonempty() {
            assert!("pineapple".try_parse("").is_err());
        }

        #[test]
        fn a_nonempty_prefix() {
            assert!("pineapple".try_parse("pine").is_err());
        }

        #[test]
        fn a_string_containing_but_not_starting_with_it() {
            assert!("pineapple".try_parse("a pineapple").is_err());
        }
    }
}

mod char_parser {
    use super::*;

    mod accepts {
        use super::*;

        #[test]
        fn a_string_starting_with_it() {
            assert!('p'.try_parse("p").is_ok());
        }
    }

    mod refuses {
        use super::*;

        #[test]
        fn an_empty_string() {
            assert!('p'.try_parse("").is_err());
        }

        #[test]
        fn a_string_containing_but_not_starting_with_it() {
            assert!('p'.try_parse("a pineapple").is_err());
        }
    }
}

mod maybe {
    use super::*;

    mod accepts {
        use super::*;

        #[test]
        fn a_string_the_input_parser_accepts() {
            let input = "pineapple pizza";

            assert!("pineapple".try_parse(input).is_ok());
            assert!("pineapple".maybe().try_parse(input).is_ok());
        }

        #[test]
        fn a_string_the_input_parser_refuses() {
            let input = "pizza";

            assert!("pineapple".try_parse(input).is_err());
            assert!("pineapple".maybe().try_parse(input).is_ok());
        }
    }
}

mod at_least_one {
    use super::*;

    mod accepts {
        use super::*;

        #[test]
        fn exactly_one() {
            assert!("pineapple".at_least_one().try_parse("pineapple").is_ok());
        }

        #[test]
        fn exactly_one_with_extra_data() {
            assert!(
                "pineapple"
                    .at_least_one()
                    .try_parse("pineapple pizza")
                    .is_ok()
            );
        }

        #[test]
        fn more_than_one() {
            assert!(
                "pineapple"
                    .at_least_one()
                    .try_parse("pineapplepineapple")
                    .is_ok()
            );
        }

        #[test]
        fn more_than_one_with_extra_data() {
            assert!(
                "pineapple"
                    .at_least_one()
                    .try_parse("pineapplepineapplepizza")
                    .is_ok()
            );
        }
    }

    mod refuses {
        use super::*;

        #[test]
        fn an_empty_string_the_input_parser_refuses() {
            // if the input parser accepts an empty string,
            // the parser enters an infinite loop
            assert!("pineapple".try_parse("").is_err());
            assert!("pineapple".at_least_one().try_parse("").is_err());
        }

        #[test]
        fn a_nonempty_string_the_input_parser_refuses() {
            let input = "pizza";

            assert!("pineapple".try_parse(input).is_err());
            assert!("pineapple".at_least_one().try_parse(input).is_err());
        }
    }
}
