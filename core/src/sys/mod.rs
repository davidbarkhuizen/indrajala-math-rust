//! What the machine is, read once a process (indrajala-ml's crate performance workplan, D5, D11):
//! sizes and counts come from it, not from a table of chips. It never changes an output's bits: it
//! only sets how a product is blocked and how many threads run it.

pub mod cpu_info;
pub mod threads;
