//! Which kernel runs at which shape, and for how long: a build for measuring only (the `census`
//! feature, never in a release; indrajala-ml's crate performance workplan, K25). A profile names a
//! function, not its operands, and which shapes dominate decides which kernel work is worth doing.
//! Without the feature [`enter`] is an empty inline function, so no code changes.
//!
//! Sites nest, and each counts its own whole time, so `conv_forward`'s includes its products: compare
//! shares within a kernel, and read a nested site's time as part of its caller's.

#[cfg(feature = "census")]
mod enabled {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Instant;

    type Key = (&'static str, usize, usize, usize);

    static COUNTS: Mutex<Option<HashMap<Key, (u64, u64)>>> = Mutex::new(None);

    /// One call's record, written when it ends.
    pub struct Site {
        key: Key,
        started: Instant,
    }

    impl Drop for Site {
        fn drop(&mut self) {
            let nanos = self.started.elapsed().as_nanos() as u64;
            let mut counts = COUNTS.lock().unwrap();
            let entry = counts.get_or_insert_with(HashMap::new).entry(self.key).or_default();
            entry.0 += 1;
            entry.1 += nanos;
        }
    }

    /// Starts one call of `kernel` at shape `m x k x n`, recorded when the result is dropped.
    pub fn enter(kernel: &'static str, m: usize, k: usize, n: usize) -> Site {
        Site {
            key: (kernel, m, k, n),
            started: Instant::now(),
        }
    }

    /// Every `(kernel, m, k, n, calls, nanoseconds)` so far.
    pub fn report() -> Vec<(&'static str, usize, usize, usize, u64, u64)> {
        let counts = COUNTS.lock().unwrap();
        counts
            .iter()
            .flatten()
            .map(|(&(kernel, m, k, n), &(calls, nanos))| (kernel, m, k, n, calls, nanos))
            .collect()
    }
}

#[cfg(feature = "census")]
pub use enabled::{enter, report};

/// Nothing to record without the `census` feature.
#[cfg(not(feature = "census"))]
pub(crate) struct Site;

#[cfg(not(feature = "census"))]
#[inline(always)]
pub(crate) fn enter(_kernel: &'static str, _m: usize, _k: usize, _n: usize) -> Site {
    Site
}
