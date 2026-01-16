use crate::parsing;
use parsing::{ParseInput, ParseResult, Parser};

use super::{
    ArithmeticOperation, BinaryOperation, Call, ComparisonOperation, Expression, LogicalOperation,
    Operation, UnaryOperation, Value,
};

impl Expression {
    pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, Self> {
        let nonempty_tuple = {
            element
                .followed_by(parsing::whitespace)
                .followed_by(',')
                .and(
                    parsing::whitespace
                        .before(element)
                        .followed_by(parsing::whitespace.followed_by(',').maybe())
                        .any_amount(),
                )
                .map(|(fst, rest)| {
                    let mut result = vec![fst];
                    result.extend(rest);
                    Self::Tuple(result)
                })
                .or(element)
        };

        nonempty_tuple.try_parse(input)
    }
}

impl Call {
    pub fn parse(input: ParseInput<'_>) -> ParseResult<'_, Self> {
        let (f, input) = primary.try_parse(input)?;
        let ((initial_args, remaining_arg_lists), input) = parsing::whitespace
            .before(args)
            .at_least_one()
            .try_parse(input)?;

        let result = remaining_arg_lists
            .into_iter()
            .fold(Self(Box::new(f), initial_args), |acc, args| {
                Self(Box::new(Expression::Call(acc)), args)
            });

        Ok((result, input))
    }
}

fn element(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let mul_div = {
        let mul = '*'.map_to(ArithmeticOperation::Mul);
        let div = '/'.map_to(ArithmeticOperation::Div);
        let mod_ = '%'.map_to(ArithmeticOperation::Mod);
        let fdiv = "//".map_to(ArithmeticOperation::FloorDiv);
        binop(
            pos_neg(exp),
            mul.or(fdiv)
                .or(div)
                .or(mod_)
                .map(BinaryOperation::Arithmetic),
        )
    };

    let add_sub = {
        let add = '+'.map_to(ArithmeticOperation::Add);
        let sub = '-'.map_to(ArithmeticOperation::Sub);
        binop(mul_div, add.or(sub).map(BinaryOperation::Arithmetic))
    };

    let cmp = {
        let eq = "==".map_to(ComparisonOperation::Eq);
        let ge = ">=".map_to(ComparisonOperation::Ge);
        let gt = ">".map_to(ComparisonOperation::Gt);
        let le = "<=".map_to(ComparisonOperation::Le);
        let lt = "<".map_to(ComparisonOperation::Lt);
        let ne = "!=".map_to(ComparisonOperation::Ne);

        let op = eq.or(ge).or(gt).or(le).or(lt).or(ne);

        // no chaining
        add_sub
            .followed_by(parsing::whitespace)
            .and(op)
            .followed_by(parsing::whitespace)
            .and(add_sub)
            .map(|((x, op), y)| {
                Expression::Operation(Operation::Binary(
                    BinaryOperation::Comparison(op),
                    Box::new(x),
                    Box::new(y),
                ))
            })
            .or(add_sub)
    };

    let and = {
        let op = "and"
            .followed_by(parsing::identifier_boundary)
            .map_to(LogicalOperation::And);
        binop(cmp, op.map(BinaryOperation::Logical))
    };

    let or = {
        let op = "or"
            .followed_by(parsing::identifier_boundary)
            .map_to(LogicalOperation::Or);
        binop(and, op.map(BinaryOperation::Logical))
    };

    or.try_parse(input)
}

fn binop<'a>(
    atom: impl Parser<'a, Expression>,
    operation: impl Parser<'a, BinaryOperation>,
) -> impl Parser<'a, Expression> {
    atom.and(
        parsing::whitespace
            .before(operation)
            .followed_by(parsing::whitespace)
            .and(atom)
            .any_amount(),
    )
    .map(|(initial_term, terms)| {
        terms.into_iter().fold(initial_term, |acc, (op, term)| {
            Expression::Operation(Operation::Binary(op, Box::new(acc), Box::new(term)))
        })
    })
}

fn pos_neg<'a>(atom: impl Parser<'a, Expression>) -> impl Parser<'a, Expression> {
    let pos = '+'.map_to(UnaryOperation::Pos);
    let neg = '-'.map_to(UnaryOperation::Neg);

    neg.or(pos)
        .followed_by(parsing::whitespace)
        .any_amount()
        .and(atom)
        .map(|(ops, base)| {
            ops.into_iter().rev().fold(base, |acc, op| {
                Expression::Operation(Operation::Unary(op, Box::new(acc)))
            })
        })
}

fn exp(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    let call = Call::parse.map(Expression::Call).or(primary);

    // right associative
    call.followed_by(parsing::whitespace)
        .followed_by("**")
        .followed_by(parsing::whitespace)
        .and(pos_neg(exp))
        .map(|(x, y)| {
            Expression::Operation(Operation::Binary(
                BinaryOperation::Arithmetic(ArithmeticOperation::Exp),
                Box::new(x),
                Box::new(y),
            ))
        })
        .or(pos_neg(call))
        .try_parse(input)
}

fn primary(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    enclosed.or(atom).try_parse(input)
}

fn enclosed(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    parsing::open_paren
        .before(parsing::whitespace)
        .before(
            Expression::parse
                .followed_by(parsing::whitespace)
                .or(parsing::nothing.map(|()| Expression::Tuple(Vec::new()))),
        )
        .followed_by(parsing::close_paren)
        .try_parse(input)
}

fn atom(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    Value::parse
        .map(Expression::Literal)
        .or(identifier)
        .try_parse(input)
}

fn args(input: ParseInput<'_>) -> ParseResult<'_, Vec<Expression>> {
    parsing::open_paren
        .before(
            parsing::whitespace
                .before(element)
                .followed_by(parsing::whitespace.followed_by(',').maybe())
                .any_amount(),
        )
        .followed_by(parsing::whitespace)
        .followed_by(parsing::close_paren)
        .try_parse(input)
}

fn identifier(input: ParseInput<'_>) -> ParseResult<'_, Expression> {
    parsing::identifier_string
        .map(Expression::Identifier)
        .try_parse(input)
}
