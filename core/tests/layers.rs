//! The core's layering (indrajala-ml's crate structure workplan, D1): a module uses only its own
//! layer and those below it, so a kernel two layers share lives in the lower one. Reads the
//! sources and fails on any `crate::` path in code (not comments) that points up.

use std::fs;
use std::path::{Path, PathBuf};

/// A top-level module's layer; `None` for one the test doesn't know, which fails it, so a new
/// module is placed on purpose.
fn layer(module: &str) -> Option<u8> {
    match module {
        "array" | "census" | "error" | "Array" | "Shape" | "Error" => Some(0),
        "num" | "sys" => Some(1),
        "ops" | "linalg" | "random" => Some(2),
        "nn" => Some(3),
        "program" | "data" => Some(4),
        _ => None,
    }
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The identifier after each `crate::` in `line`.
fn crate_paths(line: &str) -> Vec<&str> {
    line.match_indices("crate::")
        .filter(|(i, _)| *i == 0 || !line.as_bytes()[i - 1].is_ascii_alphanumeric() && line.as_bytes()[i - 1] != b'_')
        .map(|(i, m)| {
            let rest = &line[i + m.len()..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            &rest[..end]
        })
        .collect()
}

#[test]
fn every_module_uses_only_its_own_layer_and_those_below() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&src, &mut files);
    let mut upward = Vec::new();
    for file in files {
        let relative = file.strip_prefix(&src).unwrap();
        let top = relative.components().next().unwrap().as_os_str().to_str().unwrap();
        let top = top.strip_suffix(".rs").unwrap_or(top);
        if top == "lib" {
            continue;
        }
        let own = layer(top).unwrap_or_else(|| panic!("{top} (src/{}) has no layer", relative.display()));
        for (n, line) in fs::read_to_string(&file).unwrap().lines().enumerate() {
            let code = line.split("//").next().unwrap();
            for target in crate_paths(code) {
                let theirs = layer(target).unwrap_or_else(|| panic!("crate::{target} has no layer"));
                if theirs > own {
                    upward.push(format!(
                        "src/{}:{}: {top} uses crate::{target}",
                        relative.display(),
                        n + 1
                    ));
                }
            }
        }
    }
    assert!(upward.is_empty(), "upward imports:\n{}", upward.join("\n"));
}

#[test]
fn the_check_sees_a_path_and_skips_a_longer_name() {
    assert_eq!(crate_paths("use crate::nn::dense;"), vec!["nn"]);
    assert_eq!(crate_paths("x = indrajala_crate::y;"), Vec::<&str>::new());
}
