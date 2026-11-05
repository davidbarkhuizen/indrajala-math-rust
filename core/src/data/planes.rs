//! One-hot planes: small integer ids, one per cell, to the channel-major one-hot input a conv
//! layer reads (the NES plan's D13, in indrajala-research). The ids cross the boundary as bytes,
//! a byte a cell, and the float planes are built here, so nothing the size of the input is
//! converted from Python. The op knows no game: which id means what is its caller's.

use crate::array::Array;
use crate::Error;

/// `ids` holds examples of `cells` ids each, row-major. Returns `(n, channels * cells)`, `n =
/// ids.len() / cells`: example `e`'s row holds `channels` planes of `cells`, channel-major (flat
/// index `c * cells + cell`, conv's layout), `1.0` where `ids[e * cells + cell] == c` and `0.0`
/// elsewhere. An id of `channels` or more is an error, as is a length that isn't whole examples.
pub fn one_hot_planes(ids: &[u8], cells: usize, channels: usize) -> Result<Array, Error> {
    if cells == 0 || channels == 0 {
        return Err(Error::invalid(format!(
            "one_hot_planes: cells and channels must be at least 1; got {cells} and {channels}"
        )));
    }
    if ids.is_empty() || !ids.len().is_multiple_of(cells) {
        return Err(Error::invalid(format!(
            "one_hot_planes: {} ids are not a whole number of examples of {cells} cells",
            ids.len()
        )));
    }
    if let Some(&id) = ids.iter().find(|&&id| id as usize >= channels) {
        return Err(Error::invalid(format!(
            "one_hot_planes: id {id} needs more than {channels} channels"
        )));
    }
    let n = ids.len() / cells;
    let width = channels * cells;
    let mut planes = vec![0.0; n * width];
    for (example_ids, row) in ids.chunks_exact(cells).zip(planes.chunks_exact_mut(width)) {
        for (cell, &id) in example_ids.iter().enumerate() {
            row[id as usize * cells + cell] = 1.0;
        }
    }
    Ok(Array::from_matrix(planes, n, width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_cell_is_one_in_its_ids_plane_only() {
        // 2 examples of 3 cells, 4 channels
        let a = one_hot_planes(&[0, 3, 1, 2, 2, 0], 3, 4).unwrap();
        let row = |e: usize| a.data()[e * 12..(e + 1) * 12].to_vec();
        #[rustfmt::skip]
        let first = vec![1.0, 0.0, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0,  0.0, 1.0, 0.0];
        #[rustfmt::skip]
        let second = vec![0.0, 0.0, 1.0,  0.0, 0.0, 0.0,  1.0, 1.0, 0.0,  0.0, 0.0, 0.0];
        assert_eq!(row(0), first);
        assert_eq!(row(1), second);
    }

    #[test]
    fn refuses_an_id_past_the_channels_and_partial_examples() {
        assert!(one_hot_planes(&[0, 4], 2, 4).is_err());
        assert!(one_hot_planes(&[0, 1, 2], 2, 4).is_err());
        assert!(one_hot_planes(&[], 2, 4).is_err());
        assert!(one_hot_planes(&[0], 0, 4).is_err());
    }
}
