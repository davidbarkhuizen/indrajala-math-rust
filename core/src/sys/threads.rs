//! One thread budget a process (indrajala-ml's crate performance workplan, D4): the most threads the
//! crate uses for one call. `set_threads`, else `INDRAJALA_THREADS`, else the machine's physical cores,
//! at most 8 until the threshold is re-swept. A launcher running several processes sets 1 a worker, so
//! they don't oversubscribe the host. It can't change an output's bits: work is split by output.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use super::cpu_info::CpuInfo;

/// The most threads a default budget gives. Above it nothing has been measured to gain.
const MAX_DEFAULT_THREADS: usize = 8;

static SET_THREADS: AtomicUsize = AtomicUsize::new(0);

/// This process's budget from now on; 0 goes back to the environment's or the machine's. A count past
/// the machine's cores is taken as given.
pub fn set_threads(threads: usize) {
    SET_THREADS.store(threads, Ordering::Relaxed);
}

/// This process's budget, at least 1.
pub fn threads() -> usize {
    match SET_THREADS.load(Ordering::Relaxed) {
        0 => default_threads(),
        set => set,
    }
}

/// `INDRAJALA_THREADS` read once, else the machine's figure.
fn default_threads() -> usize {
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        let allowed = std::thread::available_parallelism().map_or(1, |n| n.get());
        from_environment(std::env::var("INDRAJALA_THREADS").ok().as_deref())
            .unwrap_or_else(|| machine_threads(CpuInfo::get().physical_cores, allowed))
    })
}

/// The variable's count. Unset, 0 or unreadable says nothing, as an unset one does; `threads()` shows
/// a launcher what took effect.
fn from_environment(value: Option<&str>) -> Option<usize> {
    value?.trim().parse().ok().filter(|&threads| threads > 0)
}

/// Physical cores, as SMT's second threads lost on the `d` 512 epoch (the workplan's stage 8), within
/// the CPUs this process may run on (`allowed`: its affinity and cgroup limit).
fn machine_threads(physical_cores: usize, allowed: usize) -> usize {
    physical_cores.min(allowed).clamp(1, MAX_DEFAULT_THREADS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_machine_gives_its_physical_cores_within_the_process_s_cpus() {
        assert_eq!(machine_threads(4, 8), 4); // SMT's second threads don't count
        assert_eq!(machine_threads(8, 8), 8);
        assert_eq!(machine_threads(8, 2), 2); // pinned or under a cgroup limit
        assert_eq!(machine_threads(10, 12), 8); // the cap
        assert_eq!(machine_threads(0, 0), 1);
    }

    #[test]
    fn the_environment_gives_a_positive_count_or_nothing() {
        assert_eq!(from_environment(Some("3")), Some(3));
        assert_eq!(from_environment(Some(" 12 ")), Some(12));
        for nothing in [None, Some(""), Some("0"), Some("-1"), Some("two")] {
            assert_eq!(from_environment(nothing), None);
        }
    }

    #[test]
    fn set_threads_decides_until_it_is_reset() {
        let default = threads();
        assert!(default >= 1);
        set_threads(3);
        assert_eq!(threads(), 3);
        set_threads(0);
        assert_eq!(threads(), default);
    }
}
