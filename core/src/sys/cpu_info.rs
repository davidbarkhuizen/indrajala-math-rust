//! The caches and cores, from Linux's sysfs, with safe figures where it says nothing. A cache's size
//! is per core, its size divided by the cores that share it, and the smallest across the machine's:
//! on a hybrid chip a chunk of work can land on any core, so a block must fit the smallest share (the
//! i7-1255U's P-cores have 1,280 KB of L2 each, its E-cores 2 MB across 4, so 512 KB).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

/// The figures used when sysfs doesn't say: the smallest common x86 caches, so a block fits, and one
/// core per `available_parallelism`'s logical CPU.
const DEFAULT_L1D_BYTES: usize = 32 * 1024;
const DEFAULT_L2_BYTES: usize = 256 * 1024;

/// The machine, as far as blocking and threading need it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuInfo {
    /// Physical cores: SMT siblings count once.
    pub physical_cores: usize,
    /// Logical CPUs.
    pub logical_cpus: usize,
    /// The smallest L1 data cache a core has, in bytes.
    pub l1d_per_core: usize,
    /// The smallest share of an L2 cache a core has, in bytes.
    pub l2_per_core: usize,
}

impl CpuInfo {
    /// This machine's, read once a process.
    pub fn get() -> &'static CpuInfo {
        static INFO: OnceLock<CpuInfo> = OnceLock::new();
        INFO.get_or_init(|| {
            let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
            // miri's isolation has no file system; it needs only figures that hold, and these do
            let root = if cfg!(miri) {
                Path::new("")
            } else {
                Path::new("/sys/devices/system/cpu")
            };
            CpuInfo::read(root, logical)
        })
    }

    /// The figures under a sysfs `cpu` directory, each falling back on its default when absent or
    /// unreadable (a container's masked sysfs, another operating system).
    pub fn read(root: &Path, logical_fallback: usize) -> CpuInfo {
        let mut cores = BTreeSet::new();
        let mut logical = 0;
        // (level, its shared CPU list) to its size, so a shared cache counts once
        let mut caches = BTreeMap::new();
        let mut siblings = BTreeMap::new();
        for cpu in cpu_directories(root) {
            logical += 1;
            if let Some(list) = text(&cpu.join("topology/thread_siblings_list")) {
                let group = parse_cpu_list(&list);
                if let Some(&first) = group.iter().next() {
                    for &member in &group {
                        siblings.insert(member, first);
                    }
                    cores.insert(first);
                }
            }
            for index in sub_directories(&cpu.join("cache")) {
                let kind = text(&index.join("type"));
                let level = text(&index.join("level")).and_then(|t| t.parse::<u8>().ok());
                let size = text(&index.join("size")).and_then(|t| parse_size(&t));
                let shared = text(&index.join("shared_cpu_list"));
                if let (Some(kind), Some(level), Some(size), Some(shared)) = (kind, level, size, shared) {
                    if kind != "Instruction" {
                        caches.insert((level, shared), size);
                    }
                }
            }
        }
        let per_core = |wanted: u8| {
            caches
                .iter()
                .filter(|((level, _), _)| *level == wanted)
                .map(|((_, shared), &size)| {
                    let sharing: BTreeSet<usize> = parse_cpu_list(shared)
                        .iter()
                        .map(|cpu| *siblings.get(cpu).unwrap_or(cpu))
                        .collect();
                    size / sharing.len().max(1)
                })
                .min()
        };
        let logical_cpus = if logical > 0 { logical } else { logical_fallback.max(1) };
        CpuInfo {
            physical_cores: if cores.is_empty() { logical_cpus } else { cores.len() },
            logical_cpus,
            l1d_per_core: per_core(1).unwrap_or(DEFAULT_L1D_BYTES),
            l2_per_core: per_core(2).unwrap_or(DEFAULT_L2_BYTES),
        }
    }
}

fn text(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().map(|t| t.trim().to_string())
}

fn sub_directories(dir: &Path) -> Vec<std::path::PathBuf> {
    if dir.as_os_str().is_empty() || cfg!(miri) {
        return Vec::new();
    }
    fs::read_dir(dir)
        .map(|entries| entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default()
}

/// `cpu0`, `cpu1`, ... under `root`, not `cpufreq` or `cpuidle`.
fn cpu_directories(root: &Path) -> Vec<std::path::PathBuf> {
    sub_directories(root)
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_prefix("cpu"))
                .is_some_and(|number| !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()))
        })
        .collect()
}

/// `0-3,8` to its CPUs; an unreadable part is left out.
fn parse_cpu_list(list: &str) -> BTreeSet<usize> {
    let mut cpus = BTreeSet::new();
    for part in list.split(',') {
        match part.split_once('-') {
            Some((first, last)) => {
                if let (Ok(first), Ok(last)) = (first.trim().parse::<usize>(), last.trim().parse::<usize>()) {
                    cpus.extend(first..=last);
                }
            }
            None => {
                if let Ok(cpu) = part.trim().parse() {
                    cpus.insert(cpu);
                }
            }
        }
    }
    cpus
}

/// `32K`, `2048K` or `12M` to bytes.
fn parse_size(size: &str) -> Option<usize> {
    let (digits, unit) = size.split_at(size.find(|c: char| !c.is_ascii_digit()).unwrap_or(size.len()));
    let value: usize = digits.parse().ok()?;
    match unit {
        "" => Some(value),
        "K" => Some(value * 1024),
        "M" => Some(value * 1024 * 1024),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A cache's (level, type, size, shared list), and a CPU's sibling list with its caches.
    type Cache<'a> = (u8, &'a str, &'a str, &'a str);
    type Cpu<'a> = (&'a str, &'a [Cache<'a>]);

    /// A sysfs tree of these CPUs.
    fn tree(name: &str, cpus: &[Cpu]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("indrajala-cpu-info-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for (cpu, (siblings, caches)) in cpus.iter().enumerate() {
            let dir = root.join(format!("cpu{cpu}"));
            fs::create_dir_all(dir.join("topology")).unwrap();
            fs::write(dir.join("topology/thread_siblings_list"), format!("{siblings}\n")).unwrap();
            for (index, (level, kind, size, shared)) in caches.iter().enumerate() {
                let cache = dir.join(format!("cache/index{index}"));
                fs::create_dir_all(&cache).unwrap();
                fs::write(cache.join("level"), format!("{level}\n")).unwrap();
                fs::write(cache.join("type"), format!("{kind}\n")).unwrap();
                fs::write(cache.join("size"), format!("{size}\n")).unwrap();
                fs::write(cache.join("shared_cpu_list"), format!("{shared}\n")).unwrap();
            }
        }
        fs::create_dir_all(root.join("cpufreq")).unwrap();
        root
    }

    #[test]
    fn a_plain_eight_core_machine() {
        let cpus: Vec<(String, String, String)> = (0..8)
            .map(|c| (c.to_string(), c.to_string(), "0-7".to_string()))
            .collect();
        let caches: Vec<[Cache; 3]> = cpus
            .iter()
            .map(|(_, own, all)| {
                [
                    (1, "Data", "32K", own.as_str()),
                    (2, "Unified", "256K", own.as_str()),
                    (3, "Unified", "12288K", all.as_str()),
                ]
            })
            .collect();
        let spec: Vec<Cpu> = cpus
            .iter()
            .zip(&caches)
            .map(|((s, _, _), c)| (s.as_str(), &c[..]))
            .collect();
        let root = tree("plain", &spec);
        let info = CpuInfo::read(&root, 1);
        assert_eq!((info.physical_cores, info.logical_cpus), (8, 8));
        assert_eq!((info.l1d_per_core, info.l2_per_core), (32 * 1024, 256 * 1024));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn smt_siblings_count_once_and_share_their_cache() {
        let spec: Vec<Cpu> = vec![
            (
                "0-1",
                &[
                    (1, "Data", "48K", "0-1"),
                    (2, "Unified", "512K", "0-1"),
                    (1, "Instruction", "4K", "0-1"),
                ],
            ),
            ("0-1", &[(1, "Data", "48K", "0-1"), (2, "Unified", "512K", "0-1")]),
            ("2-3", &[(1, "Data", "48K", "2-3"), (2, "Unified", "512K", "2-3")]),
            ("2-3", &[(1, "Data", "48K", "2-3"), (2, "Unified", "512K", "2-3")]),
        ];
        let root = tree("smt", &spec);
        let info = CpuInfo::read(&root, 1);
        assert_eq!((info.physical_cores, info.logical_cpus), (2, 4));
        assert_eq!((info.l1d_per_core, info.l2_per_core), (48 * 1024, 512 * 1024));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_hybrid_machine_reports_its_smallest_share() {
        // two P-cores with SMT, 1,280 KB each; four E-cores sharing 2 MB
        let p0: &[Cache] = &[(1, "Data", "48K", "0-1"), (2, "Unified", "1280K", "0-1")];
        let p1: &[Cache] = &[(1, "Data", "48K", "2-3"), (2, "Unified", "1280K", "2-3")];
        let spec: Vec<Cpu> = vec![
            ("0-1", p0),
            ("0-1", p0),
            ("2-3", p1),
            ("2-3", p1),
            ("4", &[(1, "Data", "32K", "4"), (2, "Unified", "2048K", "4-7")]),
            ("5", &[(1, "Data", "32K", "5"), (2, "Unified", "2048K", "4-7")]),
            ("6", &[(1, "Data", "32K", "6"), (2, "Unified", "2048K", "4-7")]),
            ("7", &[(1, "Data", "32K", "7"), (2, "Unified", "2048K", "4-7")]),
        ];
        let root = tree("hybrid", &spec);
        let info = CpuInfo::read(&root, 1);
        assert_eq!((info.physical_cores, info.logical_cpus), (6, 8));
        assert_eq!((info.l1d_per_core, info.l2_per_core), (32 * 1024, 512 * 1024));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn no_sysfs_falls_back_on_safe_figures() {
        let info = CpuInfo::read(Path::new("/nonexistent/sysfs"), 3);
        assert_eq!((info.physical_cores, info.logical_cpus), (3, 3));
        assert_eq!(
            (info.l1d_per_core, info.l2_per_core),
            (DEFAULT_L1D_BYTES, DEFAULT_L2_BYTES)
        );
    }

    #[test]
    fn this_machine_gives_sensible_figures() {
        let info = CpuInfo::get();
        assert!(info.physical_cores >= 1 && info.physical_cores <= info.logical_cpus);
        assert!(info.l1d_per_core >= 8 * 1024 && info.l2_per_core >= info.l1d_per_core);
    }

    #[test]
    fn the_lists_and_sizes_parse() {
        assert_eq!(
            parse_cpu_list("0-2,5").into_iter().collect::<Vec<_>>(),
            vec![0, 1, 2, 5]
        );
        assert_eq!(
            (parse_size("32K"), parse_size("12M"), parse_size("7"), parse_size("1G")),
            (Some(32768), Some(12582912), Some(7), None)
        );
    }
}
