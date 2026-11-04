//! The crate's thread pool (indrajala-ml's crate performance workplan, D3): a threaded call hands its
//! chunks to parked workers, where `std::thread::scope` starts threads on every call. Without
//! dependencies, for control over the three things that bite here: spinning (a short bound, then sleep,
//! so it doesn't steal cores from numpy as OpenBLAS's spin steals them from us), fork (Python's), and
//! chunking (claimed from a counter, so a hybrid chip's fast and slow cores even out).
//!
//! - Workers start on first use, as many as a call asks for. If the system refuses one, fewer run.
//! - A call wakes only the workers it needs, and works too. Workers it woke that haven't started when its
//!   chunks are done are told not to, so a call never waits for a sleeping worker to wake up.
//! - A chunk's panic runs through to the caller, after every other chunk has run.
//! - A call from inside a chunk, or while another thread's call has the pool, runs inline. Either way
//!   the chunks all run, once each. Callers split by output, so bits can't depend on who runs a chunk.
//! - A forked child has no workers, and its copy of the pool's locks can be in any state, so it starts
//!   a new pool: `global` keys it by the process id, and never takes a lock to find it.

use std::any::Any;
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long a worker polls for the next job before it sleeps. The workplan's stage 8: 50 µs against none
/// gains at most 8%, on back-to-back small calls, and nothing once calls are further apart than this.
const SPIN: Duration = Duration::from_micros(50);
/// How long a caller spins on workers finishing before it yields the core to them.
const WAIT_SPINS: usize = 1 << 12;
/// The most workers a pool has. Far past any machine's budget here, and a fixed size so the seats never move.
const MAX_WORKERS: usize = 64;

thread_local! {
    /// Whether this thread is running a chunk: a call from there runs inline.
    static IN_CHUNK: Cell<bool> = const { Cell::new(false) };
}

/// Marks this thread as running chunks, until dropped.
struct InChunk;

impl InChunk {
    fn enter() -> InChunk {
        IN_CHUNK.with(|flag| flag.set(true));
        InChunk
    }
}

impl Drop for InChunk {
    fn drop(&mut self) {
        IN_CHUNK.with(|flag| flag.set(false));
    }
}

/// One call. It lives on the caller's stack: the caller doesn't return until every worker that took it
/// has counted itself in `finished`, and no worker touches it after that.
struct Job<'a> {
    work: &'a (dyn Fn(usize) + Sync + 'a),
    chunks: usize,
    next: AtomicUsize,
    finished: AtomicUsize,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Job<'_> {
    /// Claims and runs chunks until there are none left, keeping the first panic.
    fn run(&self) {
        let _in_chunk = InChunk::enter();
        loop {
            let chunk = self.next.fetch_add(1, Ordering::Relaxed);
            if chunk >= self.chunks {
                return;
            }
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| (self.work)(chunk))) {
                lock(&self.panic).get_or_insert(payload);
            }
        }
    }
}

/// A chunk's panic is caught, so these locks can't be poisoned by one, and what they guard stays whole.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One worker's mailbox.
struct Seat {
    /// The job offered to this worker, and null once it's taken or withdrawn.
    job: AtomicPtr<Job<'static>>,
    sleeping: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

impl Seat {
    fn new() -> Seat {
        Seat {
            job: AtomicPtr::new(ptr::null_mut()),
            sleeping: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        }
    }

    fn offer(&self, job: *mut Job<'static>) {
        self.job.store(job, Ordering::SeqCst);
        if self.sleeping.load(Ordering::SeqCst) {
            // through the lock: a worker that saw no job holds it until it's waiting, so this can't miss it
            drop(lock(&self.lock));
            self.wake.notify_one();
        }
    }

    /// True if the worker hadn't taken the job, and now won't.
    fn withdraw(&self, job: *mut Job<'static>) -> bool {
        self.job
            .compare_exchange(job, ptr::null_mut(), Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    fn take(&self) -> Option<*mut Job<'static>> {
        if self.job.load(Ordering::SeqCst).is_null() {
            return None;
        }
        let job = self.job.swap(ptr::null_mut(), Ordering::SeqCst);
        (!job.is_null()).then_some(job)
    }
}

struct Shared {
    seats: Vec<Seat>,
    stop: AtomicBool,
}

impl Shared {
    /// The next job for this seat, polled for `SPIN`, then slept for. `None` when the pool is stopping.
    fn next_job(&self, seat: &Seat) -> Option<*mut Job<'static>> {
        loop {
            let start = Instant::now();
            while start.elapsed() < SPIN {
                if let Some(job) = seat.take() {
                    return Some(job);
                }
                if self.stop.load(Ordering::SeqCst) {
                    return None;
                }
                std::hint::spin_loop();
            }
            let mut guard = lock(&seat.lock);
            seat.sleeping.store(true, Ordering::SeqCst);
            while seat.job.load(Ordering::SeqCst).is_null() && !self.stop.load(Ordering::SeqCst) {
                guard = seat.wake.wait(guard).unwrap_or_else(PoisonError::into_inner);
            }
            seat.sleeping.store(false, Ordering::SeqCst);
            drop(guard);
            if let Some(job) = seat.take() {
                return Some(job);
            }
            if self.stop.load(Ordering::SeqCst) {
                return None;
            }
        }
    }

    fn worker(&self, index: usize) {
        let seat = &self.seats[index];
        while let Some(job) = self.next_job(seat) {
            // SAFETY: `take` won the job from its offer, so the caller counts this worker and waits for
            // `finished` before its stack frame goes. This is the last use of the pointer.
            unsafe {
                (*job).run();
                (*job).finished.fetch_add(1, Ordering::Release);
            }
        }
    }
}

/// A pool of workers. The crate's own is `global`, which lives as long as the process; dropping another
/// joins its workers.
pub struct Pool {
    shared: Arc<Shared>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    /// Workers started so far. Written only by whoever holds `busy`.
    started: AtomicUsize,
    /// Held for one call: a second caller runs inline rather than queue behind the first.
    busy: AtomicBool,
}

impl Default for Pool {
    fn default() -> Pool {
        Pool::new()
    }
}

impl Pool {
    /// A pool with no workers yet.
    pub fn new() -> Pool {
        Pool {
            shared: Arc::new(Shared {
                seats: (0..MAX_WORKERS).map(|_| Seat::new()).collect(),
                stop: AtomicBool::new(false),
            }),
            handles: Mutex::new(Vec::new()),
            started: AtomicUsize::new(0),
            busy: AtomicBool::new(false),
        }
    }

    /// Workers started so far.
    pub fn workers(&self) -> usize {
        self.started.load(Ordering::Relaxed)
    }

    /// As many of the first `wanted` workers as the system will start; how many there are now.
    fn start_workers(&self, wanted: usize) -> usize {
        let mut started = self.started.load(Ordering::Relaxed);
        while started < wanted {
            let shared = Arc::clone(&self.shared);
            let builder = std::thread::Builder::new().name("indrajala-math".into());
            let Ok(handle) = builder.spawn(move || shared.worker(started)) else {
                break;
            };
            lock(&self.handles).push(handle);
            started += 1;
            self.started.store(started, Ordering::Relaxed);
        }
        started
    }

    /// Runs `work(chunk)` for every chunk in `0..chunks`, once each, on up to `threads` threads, the
    /// caller's among them, and returns when all have ended. A chunk's panic resumes here, once all have.
    pub fn for_each_chunk(&self, chunks: usize, threads: usize, work: &(dyn Fn(usize) + Sync)) {
        let helpers = threads.min(chunks).saturating_sub(1).min(MAX_WORKERS);
        let inline = helpers == 0
            || IN_CHUNK.with(Cell::get)
            || self
                .busy
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err();
        if inline {
            (0..chunks).for_each(work);
            return;
        }
        let helpers = self.start_workers(helpers).min(helpers);
        let job = Job {
            work,
            chunks,
            next: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            panic: Mutex::new(None),
        };
        // Its lifetime is erased for the seats' sake. SAFETY: nothing dereferences it past this frame:
        // every seat's offer is withdrawn or counted in `engaged` below, and those are waited for.
        let pointer = ptr::from_ref(&job).cast::<Job<'static>>().cast_mut();
        let seats = &self.shared.seats[..helpers];
        seats.iter().for_each(|seat| seat.offer(pointer));
        job.run();
        let engaged = seats.iter().filter(|seat| !seat.withdraw(pointer)).count();
        let mut spins = 0;
        while job.finished.load(Ordering::Acquire) < engaged {
            spins += 1;
            if spins < WAIT_SPINS {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        self.busy.store(false, Ordering::Release);
        let payload = lock(&job.panic).take();
        if let Some(payload) = payload {
            panic::resume_unwind(payload);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        for seat in &self.shared.seats {
            drop(lock(&seat.lock));
            seat.wake.notify_one();
        }
        for handle in lock(&self.handles).drain(..) {
            handle.join().expect("a pool's worker catches its chunks' panics");
        }
    }
}

/// Runs `work(index, part)` for every part, once each, on the process's pool, up to `threads` threads:
/// how a threaded kernel hands out its disjoint pieces of an output, `&mut` slices among them. Each part
/// is moved out of its own lock, the pieces being few and each worth far more than that.
pub fn for_each_part<T: Send>(threads: usize, parts: Vec<T>, work: impl Fn(usize, T) + Sync) {
    let parts: Vec<Mutex<Option<T>>> = parts.into_iter().map(|part| Mutex::new(Some(part))).collect();
    global().for_each_chunk(parts.len(), threads, &|index| {
        let part = lock(&parts[index]).take().expect("a chunk runs once");
        work(index, part);
    });
}

/// The process's pool, and the process it was made in.
struct Global {
    pid: u32,
    pool: Pool,
}

static GLOBAL: AtomicPtr<Global> = AtomicPtr::new(ptr::null_mut());

/// The pool of process `pid`, made if the one held isn't this process's. The old one is left as it is,
/// never dropped: after a fork its workers don't exist, and its locks may be held forever.
fn global_of(pid: u32) -> &'static Pool {
    let current = GLOBAL.load(Ordering::Acquire);
    // SAFETY: a `Global` is leaked, so a non-null pointer stays valid.
    if let Some(global) = unsafe { current.as_ref() } {
        if global.pid == pid {
            return &global.pool;
        }
    }
    let made = Box::into_raw(Box::new(Global { pid, pool: Pool::new() }));
    match GLOBAL.compare_exchange(current, made, Ordering::AcqRel, Ordering::Acquire) {
        // SAFETY: leaked, as above.
        Ok(_) => unsafe { &(*made).pool },
        Err(_) => {
            // SAFETY: made above and never shared, and it has started no workers.
            drop(unsafe { Box::from_raw(made) });
            global_of(pid)
        }
    }
}

/// The process's pool. It runs nothing and starts no workers until a call asks it to.
pub fn global() -> &'static Pool {
    global_of(std::process::id())
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    #[test]
    fn another_process_gets_another_pool_and_keeps_it() {
        let here = std::process::id();
        let other = here.wrapping_add(1);
        let ours = ptr::from_ref(global_of(here));
        let theirs = ptr::from_ref(global_of(other));
        assert_ne!(ours, theirs);
        assert_eq!(ptr::from_ref(global_of(other)), theirs);
    }
}
