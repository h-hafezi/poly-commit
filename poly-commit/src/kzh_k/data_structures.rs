use std::sync::Arc;

use ark_ec::{pairing::Pairing, AffineRepr};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::{
    PCCommitment, PCCommitmentState, PCCommitterKey, PCUniversalParams, PCVerifierKey,
};
use crate::kzh_k::utils::Tensor;

/// Universal Parameter (SRS) later used to derive the prover and verifier keys
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""))]
pub struct KZHUniversalParams<E: Pairing> {
    // A vector of size k representing the dimensions of the tensor
    // In case of k=2, the dimensions would be [nu, mu]
    // Also, the product of the dimensions would be N: the total number of elements in the tensor
    // (the size of polynomial)
    pub(crate) dimensions: Vec<usize>,
    // h_tensors = [H1,H2,...,Hk]
    pub(crate) h_tensors: Arc<Vec<Tensor<E::G1Affine>>>,
    // Vij: i\in[d], j\in[k]
    pub(crate) v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
    // V : The G2 generator
    pub(crate) v: E::G2Affine,
    // G : The G1 generator
    pub(crate) g: E::G1Affine,
}

impl<E: Pairing> KZHUniversalParams<E> {
    /// Create a new universal parameter
    pub fn new(
        dimensions: Vec<usize>,
        h_tensors: Arc<Vec<Tensor<E::G1Affine>>>,
        v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
        v: E::G2Affine,
        g: E::G1Affine,
    ) -> Self {
        Self {
            dimensions,
            h_tensors,
            v_mat,
            v,
            g,
        }
    }
}

impl<E: Pairing> PCUniversalParams for KZHUniversalParams<E> {
    fn max_degree(&self) -> usize {
        // Only MLEs are supported
        1
    }
}

/// Prover Parameters which is derived from the SRS
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""))]
pub struct KZHCommitterKey<E: Pairing> {
    pub(crate) dimensions: Vec<usize>,
    pub(crate) h_tensors: Arc<Vec<Tensor<E::G1Affine>>>,
    pub(crate) v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
}

impl<E: Pairing> KZHCommitterKey<E> {
    /// Create a new prover parameter
    pub fn new(
        dimensions: Vec<usize>,
        h_tensors: Arc<Vec<Tensor<E::G1Affine>>>,
        v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
    ) -> Self {
        Self {
            dimensions,
            h_tensors,
            v_mat,
        }
    }
}

impl<E: Pairing> PCCommitterKey for KZHCommitterKey<E> {
    fn max_degree(&self) -> usize {
        // Only MLEs are supported
        1
    }
    fn supported_degree(&self) -> usize {
        // Only MLEs are supported
        1
    }
}


/// Verifier Parameters which is derived from the SRS, much smaller size than prover's key
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct KZHVerifierKey<E: Pairing> {
    pub(crate) dimensions: Vec<usize>,
    pub(crate) h_tensor: Arc<Tensor<E::G1Affine>>,
    pub(crate) v: E::G2Affine,
    pub(crate) v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
}

impl<E: Pairing> KZHVerifierKey<E> {
    /// Create a new verifier parameter
    pub fn new(
        dimensions: Vec<usize>,
        h_tensor: Arc<Tensor<E::G1Affine>>,
        v: E::G2Affine,
        v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
    ) -> Self {
        Self {
            dimensions,
            h_tensor,
            v,
            v_mat,
        }
    }
}

impl<E: Pairing> PCVerifierKey for KZHVerifierKey<E> {
    // Only MLEs are supported
    fn max_degree(&self) -> usize {
        1
    }
    // Only MLEs are supported
    fn supported_degree(&self) -> usize {
        1
    }
}

/// A commitment is an Affine point.
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""))]
pub struct KZHCommitment<E: Pairing> {
    /// the actual commitment is an affine point.
    pub(crate) com: E::G1Affine,
    /// number of variables
    pub(crate) nv: usize,
}

impl<E: Pairing> KZHCommitment<E> {
    /// Create a new commitment
    pub fn new(com: E::G1Affine, nv: usize) -> Self {
        Self { com, nv }
    }
}

impl<E: Pairing> PCCommitment for KZHCommitment<E> {
    #[inline]
    fn empty() -> Self {
        KZHCommitment {
            com: E::G1Affine::zero(),
            nv: 0,
        }
    }

    // The degree bound is always 1, since only multilinear polynomials are
    // supported
    fn has_degree_bound(&self) -> bool {
        true
    }
}

/// auxiliary information
#[derive(Debug, Derivative, CanonicalSerialize, CanonicalDeserialize, Clone, PartialEq, Eq)]
pub struct KZHAuxInfo<E: Pairing> {
    pub(crate) d_bool: Vec<Vec<E::G1Affine>>,
}

impl<E: Pairing> KZHAuxInfo<E> {
    /// Create a new auxiliary information
    pub fn new(d_bool: Vec<Vec<E::G1Affine>>) -> Self {
        Self { d_bool }
    }
}

impl<E: Pairing> Default for KZHAuxInfo<E> {
    fn default() -> Self {
        Self {
            d_bool: Vec::new(),
        }
    }
}

/// KZH Commitment state: the original polynomial itself, and auxiliary input
#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""))]
pub struct KZHCommitmentState<E: Pairing> {
    pub(crate) poly: DenseMultilinearExtension<E::ScalarField>,
    pub(crate) auxiliary_input: KZHAuxInfo<E>,
}

impl<E: Pairing> PCCommitmentState for KZHCommitmentState<E> {
    type Randomness = ();

    fn empty() -> Self {
        unimplemented!()
    }

    fn rand<R: RngCore>(_num_queries: usize, _has_degree_bound: bool, _num_vars: Option<usize>, _rng: &mut R) -> Self::Randomness {
        unimplemented!()
    }
}

/// proof of opening
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct KZHOpeningProof<E: Pairing> {
    pub(crate) d: Vec<Vec<E::G1Affine>>,
    pub(crate) f: DenseMultilinearExtension<E::ScalarField>,
}
