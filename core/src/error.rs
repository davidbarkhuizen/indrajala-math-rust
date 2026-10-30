use std::fmt;

/// Why an operation refused its arguments. The Python bindings raise each kind as the matching
/// Python exception, with this message.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An argument's shape or value doesn't fit the operation (Python's `ValueError`).
    InvalidArgument(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}
