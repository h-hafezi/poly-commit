use std::marker::PhantomData;
use std::sync::Arc;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::pairing::Pairing;
use ark_ec::scalar_mul::BatchMulPreprocessing;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::iterable::Iterable;
use ark_std::{One, UniformRand};
use ndarray::{ArrayD, IxDyn};
use rand::RngCore;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;

use crate::kzh_k::data_structures::{
    KZHAuxInfo, KZHCommitment, KZHCommitmentState, KZHCommitterKey, KZHOpeningProof,
    KZHUniversalParams, KZHVerifierKey,
};
use crate::kzh_k::utils::{
    build_eq_x_r, decompose_point, fix_last_variables, msm_from_scalars,
    partially_eval_dense_poly_on_bool_point, Tensor,
};
use crate::{Error, LabeledCommitment, LabeledPolynomial, PolynomialCommitment};


/// includes all the structs
pub mod data_structures;

#[cfg(test)]
mod tests;

mod utils;

/// KZH-k Polynomial Commitment Scheme
///
/// A polynomial commitment scheme based on the hardness of the discrete
/// logarithm problem in prime-order groups. The scheme is parameterized by
/// an integer `k`, which determines the trade-off between prover cost and
/// proof size.
///
/// - For the simplest case `k = 2`, the opening is extremely efficient,
///   requiring no group operations, and the proof size is `O(√N)` where
///   `N` is the size of the polynomial (i.e., the size of the Boolean hypercube).
///
/// - For general `k > 2`, the opening requires roughly `O(N^{1/2})` group
///   operations, and the proof size is `O(k · N^{1/k})`.
///   In the extreme case `k = log N`, the proof size becomes `O(log N)`.
///
/// References:
/// - KZH-fold: <https://eprint.iacr.org/2025/144>
/// - IronDict (showing how to achieve zero-knowledge using a homomorphic compiler):
///   <https://eprint.iacr.org/2025/1580>
///
/// The current implementation is not zero-knowledge, but applying the
/// homomorphic compiler from IronDict yields zero-knowledge with only
/// `O(k · N^{1/k})` randomness.
///
/// ### Future Optimizations
///
/// Due to the homomorphic structure of KZH, some methods may be implemented
/// more efficiently in the future, including: `batch_open` and `batch_check`
pub struct KZH<E: Pairing, P: MultilinearExtension<E::ScalarField>> {
    #[doc(hidden)]
    phantom: PhantomData<(E, P)>,
}

// add an associated const
impl<E: Pairing, P: MultilinearExtension<E::ScalarField>> KZH<E, P> {
    /// public constant k which is to change
    pub const K: usize = 7; // set the value you need
}


impl<E, P> PolynomialCommitment<E::ScalarField, P> for KZH<E, P>
where
    E: Pairing,
    P: MultilinearExtension<E::ScalarField>,
{
    type UniversalParams = KZHUniversalParams<E>;
    type CommitterKey = KZHCommitterKey<E>;
    type VerifierKey = KZHVerifierKey<E>;
    type Commitment = KZHCommitment<E>;
    type CommitmentState = KZHCommitmentState<E>;
    type Proof = Vec<KZHOpeningProof<E>>;
    type BatchProof = Vec<Self::Proof>;
    type Error = Error;

    fn setup<R: RngCore>(
        _max_degree: usize,
        num_vars: Option<usize>,
        rng: &mut R,
    ) -> Result<Self::UniversalParams, Self::Error> {
        // ---------------------------------------------------------------------
        // Validate input
        // ---------------------------------------------------------------------
        let num_vars = num_vars.ok_or(Error::InvalidNumberOfVariables)?;

        // ---------------------------------------------------------------------
        // Split num_vars across K dimensions
        // ---------------------------------------------------------------------
        let base = num_vars / Self::K;
        let extra = num_vars % Self::K;

        let mut dimensions: Vec<usize> = vec![base; Self::K];
        for dim in dimensions.iter_mut().take(extra) {
            *dim += 1;
        }
        // dimensions.reverse();
        println!("the dimensions is: {:?}", dimensions);

        // ---------------------------------------------------------------------
        // Public generators
        // ---------------------------------------------------------------------
        let g1_gen = E::G1::rand(rng);
        let g2_gen = E::G2::rand(rng);

        // ---------------------------------------------------------------------
        // Trapdoors μ_j : each has length 2^{d_j}
        // ---------------------------------------------------------------------
        let mu_mat: Vec<Vec<E::ScalarField>> = (0..Self::K)
            .map(|j| {
                (0..(1usize << dimensions[j]))
                    .map(|_| E::ScalarField::rand(rng))
                    .collect()
            })
            .collect();

        let mu_mat = Arc::new(mu_mat);
        let dimensions_arc = Arc::new(dimensions.clone());

        // ---------------------------------------------------------------------
        // Build H_t tensors
        // ---------------------------------------------------------------------
        let mut h_tensors = Vec::with_capacity(Self::K);

        for t in 0..Self::K {
            let dims = &dimensions_arc[t..];
            let shape: Vec<usize> = dims.iter().map(|&d| 1usize << d).collect();
            let total_len: usize = shape.iter().product();
            let axes = shape.len();

            // 1) Compute exponents exps[r_t,...,r_{k-1}] = ∏_{j=t}^{k-1} μ_j[r_j]
            let mut exps = vec![E::ScalarField::one(); total_len];

            let mut stride = 1usize;
            for axis in (0..axes).rev() {
                let size = shape[axis];
                let block = size * stride;
                let mu_j = &mu_mat[t + axis];

                #[cfg(feature = "parallel")]
                {
                    exps.par_chunks_mut(block).for_each(|chunk| {
                        for r in 0..size {
                            let mu = mu_j[r];
                            let seg = &mut chunk[r * stride..(r + 1) * stride];
                            for e in seg.iter_mut() {
                                *e *= mu;
                            }
                        }
                    });
                }

                #[cfg(not(feature = "parallel"))]
                {
                    for chunk in exps.chunks_mut(block) {
                        for r in 0..size {
                            let mu = mu_j[r];
                            let seg = &mut chunk[r * stride..(r + 1) * stride];
                            for e in seg.iter_mut() {
                                *e *= mu;
                            }
                        }
                    }
                }

                stride *= size;
            }

            // 2) Batch-multiply with G1 generator
            let tbl = BatchMulPreprocessing::new(g1_gen, total_len);
            let flat_affine = tbl.batch_mul(&exps);

            // 3) Pack into ndarray tensor
            let array = ArrayD::from_shape_vec(IxDyn(&shape), flat_affine).expect("shape must match buffer length");

            h_tensors.push(Tensor(array));
        }

        let h_tensors = Arc::new(h_tensors);

        // ---------------------------------------------------------------------
        // Build V_j matrices in G2
        // ---------------------------------------------------------------------
        let compute_v_row = |j: usize| {
            let rows = 1usize << dimensions_arc[j];
            let tbl = BatchMulPreprocessing::new(g2_gen, rows);
            let aff: Vec<E::G2Affine> = tbl.batch_mul(&mu_mat[j]);
            aff.into_iter()
                .map(<E as Pairing>::G2Prepared::from)
                .collect::<Vec<_>>()
        };

        let v_mat: Vec<Vec<<E as Pairing>::G2Prepared>> = {
            #[cfg(feature = "parallel")]
            {
                (0..Self::K).into_par_iter().map(compute_v_row).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..Self::K).map(compute_v_row).collect()
            }
        };

        let v_mat = Arc::new(v_mat);

        // ---------------------------------------------------------------------
        // Return universal parameters
        // ---------------------------------------------------------------------
        Ok(KZHUniversalParams::new(
            (*dimensions_arc).clone(),
            h_tensors,
            v_mat,
            g2_gen.into_affine(),
            g1_gen.into_affine(),
        ))
    }

    fn trim(
        pp: &Self::UniversalParams,
        _supported_degree: usize,
        _supported_hiding_bound: usize,
        _enforced_degree_bounds: Option<&[usize]>,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), Self::Error> {
        Ok((
            KZHCommitterKey::new(
                pp.dimensions.clone(),
                pp.h_tensors.clone(),
                pp.v_mat.clone(),
            ),
            KZHVerifierKey::new(
                pp.dimensions.clone(),
                pp.h_tensors[pp.dimensions.len() - 1].clone().into(),
                pp.v,
                pp.v_mat.clone(),
            ),
        ))
    }

    fn commit<'a>(
        ck: &Self::CommitterKey,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<E::ScalarField, P>>,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<
        (
            Vec<LabeledCommitment<Self::Commitment>>,
            Vec<Self::CommitmentState>,
        ),
        Self::Error,
    >
    where
        P: 'a,
    {
        let mut commitments = Vec::new();
        let mut states = Vec::new();

        // ---------------------------------------------------------------------
        // G1 bases for top-level MSM
        // ---------------------------------------------------------------------
        let bases: &[E::G1Affine] = ck.h_tensors[0]
            .as_slice_memory_order()
            .expect("H_0 tensor must have standard memory layout");

        // ---------------------------------------------------------------------
        // Process each polynomial
        // ---------------------------------------------------------------------
        for labeled_poly in polynomials {
            let label = labeled_poly.label();
            let poly = labeled_poly.polynomial();
            let num_vars = poly.num_vars();

            // -----------------------------------------------------------------
            // Sanity check: num_vars must match CK dimensions
            // -----------------------------------------------------------------
            let expected_vars: usize = ck.dimensions.iter().copied().sum();
            if num_vars != expected_vars {
                return Err(Error::InvalidNumberOfVariables);
            }

            // -----------------------------------------------------------------
            // Commitment = MSM(bases, poly_evals)
            // -----------------------------------------------------------------
            let evals = poly.to_evaluations();
            let com_point: E::G1 = msm_from_scalars::<E>(bases, evals.as_slice());

            let commitment = KZHCommitment::new(com_point.into(), num_vars);
            let labeled_commitment = LabeledCommitment::new(label.to_string(), commitment, Some(1));

            commitments.push(labeled_commitment);

            // -----------------------------------------------------------------
            // Build auxiliary structure d_bool[j]: partial MSMs by prefix blocks
            // -----------------------------------------------------------------
            let auxiliary_input = {
                let k = ck.dimensions.len();
                if k <= 1 {
                    return Err(Error::WrongParameter("k cannot be less than 2".to_string()));
                }

                let mut d_bool = Vec::with_capacity(k - 1);
                let mut prefix_vars = 0usize;

                for (j, &dim) in ck.dimensions.iter().take(k - 1).enumerate() {
                    prefix_vars += dim;

                    let prefix_size = 1usize << prefix_vars;
                    let remaining_vars = num_vars - prefix_vars;
                    let eval_len = 1usize << remaining_vars;

                    // Use H_{j+1} as required by the construction
                    let h_slice = ck.h_tensors[j + 1]
                        .as_slice_memory_order()
                        .expect("H_t tensors must be contiguous");

                    // Compute d_j[i] = <H_{j+1}[i, *], partial_eval(poly, i)>
                    let mut d_j = vec![E::G1Affine::zero(); prefix_size];

                    cfg_iter_mut!(d_j)
                        .enumerate()
                        .for_each(|(i, d_j_i)| {
                            let partial = partially_eval_dense_poly_on_bool_point(
                                evals.as_slice(),
                                i,
                                eval_len,
                            );
                            let msm_res =
                                msm_from_scalars::<E>(h_slice, partial.as_slice());
                            *d_j_i = msm_res.into_affine();
                        });

                    d_bool.push(d_j);
                }

                KZHAuxInfo { d_bool }
            };

            // -----------------------------------------------------------------
            // Save commitment state
            // -----------------------------------------------------------------
            let mle = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evals,
            );

            states.push(KZHCommitmentState {
                poly: mle,
                auxiliary_input,
            });
        }

        Ok((commitments, states))
    }

    fn open<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<E::ScalarField, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        _sponge: &mut impl CryptographicSponge,
        states: impl IntoIterator<Item = &'a Self::CommitmentState>,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Self::Error>
    where
        P: 'a,
        Self::CommitmentState: 'a,
        Self::Commitment: 'a,
    {
        let mut proofs = Vec::new();

        let mut polys = labeled_polynomials.into_iter();
        let mut comms = commitments.into_iter();
        let mut st_iter = states.into_iter();

        // Precompute once
        let total_vars: usize = ck.dimensions.iter().sum();
        let dims = ck.dimensions.as_slice();
        let k = dims.len();
        let decomposed_point = decompose_point(dims, point);

        while let (Some(l_poly), Some(l_com), Some(state)) =
            (polys.next(), comms.next(), st_iter.next())
        {
            // ────────────────────────────────────────
            // Label consistency
            // ────────────────────────────────────────
            if l_poly.label() != l_com.label() {
                return Err(Error::MismatchedLabels {
                    commitment_label: l_com.label().to_string(),
                    polynomial_label: l_poly.label().to_string(),
                });
            }

            let poly = l_poly.polynomial();

            // ────────────────────────────────────────
            // Number of variables check
            // ────────────────────────────────────────
            if poly.num_vars() != total_vars {
                return Err(Error::MismatchedNumVars {
                    poly_nv: poly.num_vars(),
                    point_nv: total_vars,
                });
            }

            // Start with the full polynomial as a DME
            let mut partial_poly = DenseMultilinearExtension::from_evaluations_vec(
                poly.num_vars(),
                poly.to_evaluations(),
            );

            let mut d = Vec::with_capacity(k - 1);

            // ────────────────────────────────────────
            // Compute d_j values for j in 0 .. k-2
            // ────────────────────────────────────────
            for (j, point_part) in decomposed_point.iter().take(k - 1).enumerate() {
                let evals = partial_poly.to_evaluations();
                let block_vars = dims[j];

                let num_chunks = 1 << block_vars; // 2^{dim_j}
                let chunk_len = evals.len() / num_chunks; // 2^(n - prefix)

                if chunk_len <= 0 {
                    return Err(Error::WrongParameter("chunk length should be a positive number".to_string()));
                }
                if evals.len() % num_chunks != 0 {
                    return Err(Error::WrongParameter("eval array should be dividable by chunk_len".to_string()));
                }

                let h_slice = ck.h_tensors[j + 1]
                    .as_slice_memory_order()
                    .expect("H_t must be contiguous");

                // Build d_j
                let dj: Vec<E::G1Affine> = cfg_into_iter!(0..num_chunks)
                    .map(|i| {
                        let offset = i * chunk_len;
                        let chunk = &evals[offset..offset + chunk_len];
                        msm_from_scalars::<E>(h_slice, chunk).into_affine()
                    })
                    .collect();

                d.push(dj);

                // Update the partially fixed polynomial
                partial_poly = fix_last_variables(&partial_poly, point_part.as_slice())?;
            }

            // ────────────────────────────────────────
            // Push final opening proof for this poly
            // ────────────────────────────────────────
            proofs.push(KZHOpeningProof {
                d,
                f: partial_poly,
            });
        }

        Ok(proofs)
    }

    fn check<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = E::ScalarField>,
        proof: &Self::Proof,
        _sponge: &mut impl CryptographicSponge,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        let mut commitments = commitments.into_iter();
        let mut values = values.into_iter();
        let mut proofs = proof.iter();

        // Precompute the decomposition once
        let dims = vk.dimensions.as_slice();
        let decomposed_point = decompose_point(dims, point);
        let k = dims.len();

        while let (Some(com), Some(pf), Some(value)) =
            (commitments.next(), proofs.next(), values.next())
        {
            let mut cj = com.commitment().com;

            // ───────────────────────────────
            // Check c_0 .. c_{k-2}
            // ───────────────────────────────
            for (j, point_part) in decomposed_point.iter().take(k - 1).enumerate() {
                let cj_prepared = <E as Pairing>::G1Prepared::from(cj);
                let v_prepared = <E as Pairing>::G2Prepared::from(vk.v);

                // Prepare pairing inputs
                let g1_terms: Vec<_> = pf.d[j]
                    .iter()
                    .copied()
                    .map(<E as Pairing>::G1Prepared::from)
                    .collect();

                let g2_terms: Vec<_> = vk.v_mat[j].clone();

                // Pairing consistency check
                let lhs = E::multi_pairing(g1_terms, g2_terms);
                let rhs = E::pairing(cj_prepared, v_prepared);

                // if lhs != rhs { return Ok(false); }
                assert_eq!(lhs, rhs);


                // Update cj using the eq polynomial
                let eq_poly = build_eq_x_r(point_part)?;
                cj = msm_from_scalars::<E>(&pf.d[j], &eq_poly.evaluations).into_affine();
            }

            // ───────────────────────────────
            // Check c_{k-1}
            // ───────────────────────────────

            let alleged_last_cj = E::G1::msm(
                vk.h_tensor.as_slice_memory_order().unwrap(),
                &pf.f.to_evaluations(),
            )
                .unwrap()
                .into_affine();

            // if cj != alleged_last_cj { return Ok(false); }
            assert_eq!(cj, alleged_last_cj);

            // ───────────────────────────────
            // Final evaluation check
            // ───────────────────────────────

            let eval = fix_last_variables(&pf.f, &decomposed_point[k-1])?[0];
            /*if eval != value {
                return Ok(false);
            }
             */
            assert_eq!(eval, value);
        }

        Ok(true)
    }
}

