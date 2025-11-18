use std::convert::TryFrom;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use ark_ec::VariableBaseMSM;
use ark_ff::{Field, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid,
    Validate, Write,
};
use ndarray::{ArrayD, IxDyn};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use crate::Error;

/// Compute MSM(bases, scalars) by first converting all scalars to BigInt.
pub(crate) fn msm_from_scalars<E: ark_ec::pairing::Pairing>(bases: &[E::G1Affine], scalars: &[E::ScalarField]) -> E::G1 {
    // Convert scalars → BigInts
    let scalars_bigint = ark_std::cfg_iter!(scalars)
        .map(|s| s.into_bigint())
        .collect::<Vec<_>>();

    // Run MSM
    <E::G1 as VariableBaseMSM>::msm_bigint(bases, &scalars_bigint)
}

pub(crate) fn partially_eval_dense_poly_on_bool_point<F: Field>(evals: &[F], index: usize, n: usize) -> Vec<F> {
    // This compiles down to a highly optimized memory
    evals[n * index..(n * index + n)].to_vec()
}

pub(crate) fn decompose_point<T: Clone>(dimensions: &[usize], point: &[T]) -> Vec<Vec<T>> {
    let mut decomposed = Vec::new();
    let mut start = 0;
    for &dim in dimensions {
        let end = start + dim;
        decomposed.push(point[start..end].to_vec());
        start = end;
    }
    // this has to be reversed because of how Arkworks partially evaluation works from the end
    decomposed.reverse();

    decomposed
}

pub(crate) fn fix_last_variables<F: PrimeField>(poly: &DenseMultilinearExtension<F>, partial_point: &[F]) -> Result<DenseMultilinearExtension<F>, Error> {
    if partial_point.len() > poly.num_vars {
        return Err(Error::IncorrectInputLength("invalid size of partial point".to_string()));
    }

    let nu = partial_point.len();
    let mu = poly.num_vars - nu;

    let mut current_evals = poly.evaluations.to_vec();

    // Evaluate single variable of partial point from right to left (MSB to LSB).
    for (i, point) in partial_point.iter().rev().enumerate() {
        current_evals = fix_last_variable_helper(&current_evals, poly.num_vars - i, point);
    }

    Ok(DenseMultilinearExtension::<F>::from_evaluations_slice(mu, &current_evals[..1usize << mu]))
}

fn fix_last_variable_helper<F: Field>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let half_len = 1usize << (nv - 1);
    let mut res = vec![F::zero(); half_len];

    // evaluate single variable of partial point from left to right
    #[cfg(not(feature = "parallel"))]
    for b in 0..half_len {
        res[b] = data[b] + (data[b + half_len] - data[b]) * point;
    }

    #[cfg(feature = "parallel")]
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i] + (data[i + half_len] - data[i]) * point;
    });

    res
}


/// Build the multilinear polynomial eq(x, r), defined as:
///     eq(x, r) = ∏ᵢ (xᵢ * rᵢ + (1 - xᵢ)(1 - rᵢ)),
/// evaluated over all x ∈ {0,1}ⁿ (so the output has 2ⁿ entries).
///
/// Conceptually, at step i we extend the current evaluation vector by:
///   - multiplying each value by (1 - rᵢ) for xᵢ = 0
///   - multiplying each value by rᵢ       for xᵢ = 1
///
/// This builds the full evaluation table recursively.
pub(crate) fn build_eq_x_r<F: PrimeField>(
    r: &[F],
) -> Result<Arc<DenseMultilinearExtension<F>>, Error> {
    if r.is_empty() {
        return Err(Error::DegreeIsZero)
    }

    // Build all 2^n evaluations of eq(x, r)
    let mut evals = Vec::new();
    build_eq_x_r_recursive(r, &mut evals);

    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(r.len(), evals)))
}

/// Recursive construction of eq(x, r) evaluation table.
/// For r of length n, produces a vector of length 2ⁿ.
/// At each step, the table doubles:
///   new[2i]   = (1 - r₀) * old[i]
///   new[2i+1] = r₀       * old[i]
fn build_eq_x_r_recursive<F: PrimeField>(r: &[F], buf: &mut Vec<F>) {
    if r.len() == 1 {
        // Base case: [1 - r₀, r₀]
        buf.push(F::one() - r[0]);
        buf.push(r[0]);
    } else {
        // Recursively fill buf with evaluations ignoring r₀
        build_eq_x_r_recursive(&r[1..], buf);

        // Now extend using r₀
        let mut res = vec![F::zero(); buf.len() * 2];

        cfg_iter_mut!(res).enumerate().for_each(|(i, val)| {
            let bi = buf[i >> 1];
            let tmp = r[0] * bi;
            if i & 1 == 0 {
                *val = bi - tmp; // (1 - r₀) * bi
            } else {
                *val = tmp;      // r₀ * bi
            }
        });
        *buf = res;
    }
}


///////////////// Tensor and implementation ///////////////////

/// Local newtype wrapper around `ndarray::ArrayD<T>` so we can implement
/// `CanonicalSerialize`/`CanonicalDeserialize` without violating the orphan
/// rules.
#[derive(Clone, Debug)]
pub struct Tensor<T: Sized>(pub ArrayD<T>);

impl<T> From<ArrayD<T>> for Tensor<T> {
    fn from(a: ArrayD<T>) -> Self {
        Tensor(a)
    }
}
impl<T> From<Tensor<T>> for ArrayD<T> {
    fn from(w: Tensor<T>) -> Self {
        w.0
    }
}

impl<T> Deref for Tensor<T> {
    type Target = ArrayD<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn product_u64(shape: &[usize]) -> Result<u64, SerializationError> {
    let mut acc: u128 = 1;
    for &d in shape {
        acc = acc
            .checked_mul(d as u128)
            .ok_or(SerializationError::InvalidData)?;
    }
    u64::try_from(acc).map_err(|_| SerializationError::InvalidData)
}

/// Iterator to walk all indices in row-major order for a given shape.
struct RowMajorIx {
    idx: Vec<usize>,
    shape: Vec<usize>,
    done: bool,
}

impl RowMajorIx {
    fn new(shape: &[usize]) -> Self {
        let k = shape.len();
        let done = shape.contains(&0);
        Self {
            idx: vec![0; k],
            shape: shape.to_vec(),
            done,
        }
    }
}

impl Iterator for RowMajorIx {
    type Item = IxDyn;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let out = IxDyn(&self.idx);
        for ax in (0..self.shape.len()).rev() {
            self.idx[ax] += 1;
            if self.idx[ax] < self.shape[ax] {
                break;
            } else {
                self.idx[ax] = 0;
                if ax == 0 {
                    self.done = true;
                }
            }
        }
        Some(out)
    }
}

impl<T: CanonicalSerialize> CanonicalSerialize for Tensor<T> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut w: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // rank
        let rank = u32::try_from(self.ndim()).map_err(|_| SerializationError::InvalidData)?;
        rank.serialize_with_mode(&mut w, compress)?;
        // shape
        for &d in self.shape() {
            let d64 = u64::try_from(d).map_err(|_| SerializationError::InvalidData)?;
            d64.serialize_with_mode(&mut w, compress)?;
        }
        // element count
        let n = product_u64(self.shape())?;
        n.serialize_with_mode(&mut w, compress)?;
        // elements in row-major order
        let shape = self.shape().to_vec();
        if self.is_standard_layout() {
            if let Some(slice) = self.as_slice_memory_order() {
                for t in slice {
                    t.serialize_with_mode(&mut w, compress)?;
                }
                return Ok(());
            }
        }
        for ix in RowMajorIx::new(&shape) {
            self[ix].serialize_with_mode(&mut w, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut sz = 0usize;
        sz += u32::default().serialized_size(compress);
        sz += self.shape().len() * u64::default().serialized_size(compress);
        sz += u64::default().serialized_size(compress);
        if self.is_standard_layout() {
            if let Some(slice) = self.as_slice_memory_order() {
                return sz
                    + slice
                    .iter()
                    .map(|t| t.serialized_size(compress))
                    .sum::<usize>();
            }
        }
        let shape = self.shape().to_vec();
        sz + RowMajorIx::new(&shape)
            .map(|ix| self[ix].serialized_size(compress))
            .sum::<usize>()
    }
}

impl<T: Valid> Valid for Tensor<T> {
    fn check(&self) -> Result<(), SerializationError> {
        // Check each element
        if self.is_standard_layout() {
            if let Some(slice) = self.as_slice_memory_order() {
                for t in slice {
                    t.check()?;
                }
                return Ok(());
            }
        }

        let shape = self.shape().to_vec();
        for ix in RowMajorIx::new(&shape) {
            self[ix].check()?;
        }
        Ok(())
    }
}

impl<T: Valid + CanonicalDeserialize> CanonicalDeserialize for Tensor<T> {
    fn deserialize_with_mode<R: Read>(
        mut r: R,
        compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let k = u32::deserialize_with_mode(&mut r, compress, Validate::No)?;
        let k = usize::try_from(k).map_err(|_| SerializationError::InvalidData)?;
        // shape
        let mut shape = Vec::with_capacity(k);
        for _ in 0..k {
            let d = u64::deserialize_with_mode(&mut r, compress, Validate::No)?;
            shape.push(usize::try_from(d).map_err(|_| SerializationError::InvalidData)?);
        }
        // element count check
        let n_hdr = u64::deserialize_with_mode(&mut r, compress, Validate::No)?;
        let n_calc = product_u64(&shape)?;
        if n_hdr != n_calc {
            return Err(SerializationError::InvalidData);
        }
        let n = usize::try_from(n_calc).map_err(|_| SerializationError::InvalidData)?;
        // elements in row-major order
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(T::deserialize_with_mode(&mut r, compress, Validate::No)?);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|_| SerializationError::InvalidData)?;
        Ok(Tensor(arr))
    }
}
