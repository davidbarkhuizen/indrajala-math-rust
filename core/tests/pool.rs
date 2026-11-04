//! The pool (`sys::pool`): every chunk runs once at every chunk and thread count, whoever calls it,
//! from inside a chunk too; a panic reaches the caller; a forked child gets working threads. Runs under
//! miri, with fewer rounds, on pools it owns, so it can join them.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use indrajala_math::sys::pool::Pool;

/// Each chunk's count of runs.
fn runs_of(pool: &Pool, chunks: usize, threads: usize) -> Vec<usize> {
    let counts: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
    pool.for_each_chunk(chunks, threads, &|chunk| {
        counts[chunk].fetch_add(1, Ordering::Relaxed);
    });
    counts.into_iter().map(AtomicUsize::into_inner).collect()
}

#[test]
fn every_chunk_runs_once_at_every_count() {
    let pool = Pool::new();
    for chunks in [0, 1, 2, 3, 8, 33] {
        for threads in [0, 1, 2, 3, 8] {
            assert_eq!(
                runs_of(&pool, chunks, threads),
                vec![1; chunks],
                "{chunks} chunks, {threads} threads"
            );
        }
    }
}

#[test]
fn a_stress_of_random_counts_runs_every_chunk_once() {
    let rounds = if cfg!(miri) { 30 } else { 20_000 };
    let pool = Pool::new();
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..rounds {
        state = state.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(1);
        let chunks = (state >> 40) as usize % 40;
        let threads = (state >> 50) as usize % 9;
        assert_eq!(
            runs_of(&pool, chunks, threads),
            vec![1; chunks],
            "{chunks} chunks, {threads} threads"
        );
    }
}

#[test]
fn calls_that_sleep_between_them_find_the_workers_again() {
    let pool = Pool::new();
    for _ in 0..if cfg!(miri) { 3 } else { 20 } {
        assert_eq!(runs_of(&pool, 8, 4), vec![1; 8]);
        std::thread::sleep(std::time::Duration::from_millis(2)); // past the spin: they've parked
    }
}

#[test]
fn it_starts_only_the_workers_a_call_can_use() {
    let pool = Pool::new();
    assert_eq!(pool.workers(), 0);
    pool.for_each_chunk(1, 8, &|_| {}); // one chunk: the caller's
    pool.for_each_chunk(8, 1, &|_| {}); // one thread: the caller's
    assert_eq!(pool.workers(), 0);
    pool.for_each_chunk(3, 8, &|_| {}); // the caller and two
    assert_eq!(pool.workers(), 2);
    pool.for_each_chunk(8, 4, &|_| {});
    assert_eq!(pool.workers(), 3);
    pool.for_each_chunk(8, 2, &|_| {}); // fewer than it has: none go
    assert_eq!(pool.workers(), 3);
}

#[test]
fn a_chunk_can_borrow_the_caller_s_data_and_write_its_own_part() {
    let pool = Pool::new();
    let input: Vec<usize> = (0..64).collect();
    let output: Vec<Mutex<usize>> = (0..64).map(|_| Mutex::new(0)).collect();
    pool.for_each_chunk(64, 4, &|chunk| *output[chunk].lock().unwrap() = 2 * input[chunk]);
    let doubled: Vec<usize> = output.into_iter().map(|cell| cell.into_inner().unwrap()).collect();
    assert_eq!(doubled, input.iter().map(|value| 2 * value).collect::<Vec<_>>());
}

#[test]
fn a_call_from_inside_a_chunk_runs_on_that_thread() {
    let pool = Pool::new();
    let inner = AtomicUsize::new(0);
    pool.for_each_chunk(8, 4, &|_| {
        let outer_thread = std::thread::current().id();
        pool.for_each_chunk(5, 4, &|_| {
            assert_eq!(std::thread::current().id(), outer_thread);
            inner.fetch_add(1, Ordering::Relaxed);
        });
    });
    assert_eq!(inner.into_inner(), 8 * 5);
}

#[test]
fn callers_at_once_each_get_every_chunk_once() {
    let rounds = if cfg!(miri) { 5 } else { 2_000 };
    let pool = Pool::new();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..rounds {
                    assert_eq!(runs_of(&pool, 16, 4), vec![1; 16]);
                }
            });
        }
    });
}

#[test]
fn a_chunk_s_panic_reaches_the_caller_after_the_others_run_and_the_pool_still_works() {
    let pool = Pool::new();
    let ran = AtomicUsize::new(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.for_each_chunk(16, 4, &|chunk| {
            ran.fetch_add(1, Ordering::Relaxed);
            assert_ne!(chunk, 5, "chunk five");
        });
    }));
    let message = result.unwrap_err();
    assert!(message.downcast_ref::<String>().unwrap().contains("chunk five"));
    assert_eq!(ran.into_inner(), 16);
    assert_eq!(runs_of(&pool, 16, 4), vec![1; 16]);
}

#[test]
fn several_threads_really_run_a_call() {
    let pool = Pool::new();
    // each chunk waits for another thread to arrive, so this hangs if the pool runs them in turn
    let arrived = AtomicUsize::new(0);
    let threads = Mutex::new(HashSet::new());
    pool.for_each_chunk(2, 2, &|_| {
        arrived.fetch_add(1, Ordering::SeqCst);
        threads.lock().unwrap().insert(std::thread::current().id());
        while arrived.load(Ordering::SeqCst) < 2 {
            std::hint::spin_loop();
        }
    });
    assert_eq!(threads.into_inner().unwrap().len(), 2);
}

#[cfg(all(unix, not(miri)))]
mod fork {
    use super::*;
    use indrajala_math::sys::pool::global;

    unsafe extern "C" {
        fn fork() -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        fn _exit(status: i32) -> !;
        fn alarm(seconds: u32) -> u32;
    }

    /// The parent has used the global pool, so its workers exist; a child inherits none of them. A pool
    /// that kept them would hang, which the alarm turns into a failed status.
    #[test]
    fn a_forked_child_gets_its_own_workers() {
        let parent = global();
        assert_eq!(runs_of(parent, 16, 4), vec![1; 16]);
        // SAFETY: the child calls only what this test is about, then `_exit`, as nothing else of the
        // test harness's may run there.
        unsafe {
            let child = fork();
            assert!(child >= 0);
            if child == 0 {
                alarm(60);
                let pool = global();
                let same = std::ptr::eq(pool, parent);
                let right = runs_of(pool, 16, 4) == vec![1; 16];
                _exit(if !same && right && pool.workers() == 3 { 0 } else { 1 });
            }
            let mut status = -1;
            assert_eq!(waitpid(child, &raw mut status, 0), child);
            assert_eq!(status, 0, "the child's status");
        }
        assert_eq!(runs_of(global(), 16, 4), vec![1; 16]);
    }
}
