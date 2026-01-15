use crate::parsing::ParseError;

#[derive(Clone, Copy, Debug, Default)]
pub enum VariableState {
    Global,
    Local,
    #[default]
    Unmarked,
}

impl VariableState {
    pub const fn after_assignment(self) -> Self {
        match self {
            Self::Global => Self::Global,
            Self::Local | Self::Unmarked => Self::Local,
        }
    }

    pub const fn after_global(self) -> Result<Self, ParseError> {
        match self {
            Self::Global | Self::Unmarked => Ok(Self::Global),
            Self::Local => Err(ParseError),
        }
    }
}
