//! numpy's two generator families, bit for bit: the legacy global `np.random` (MT19937) and
//! `default_rng` (PCG64 under a `SeedSequence`).

pub mod generator;
pub mod legacy;
