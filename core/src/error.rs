use std::fmt;

/// Why an operation refused its arguments. The Python bindings raise each kind as the matching
/// Python exception, with this message.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An argument's shape or value doesn't fit the operation (Python's `ValueError`).
    InvalidArgument(String),
    /// A value is out of the operation's range (Python's `OverflowError`).
    Overflow(String),
}

impl Error {
    /// An [`Error::InvalidArgument`] with `message`.
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Error::InvalidArgument(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(message) | Error::Overflow(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}
