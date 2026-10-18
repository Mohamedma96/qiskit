// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::error::{MathOpError, check_axis, check_dtype};
use super::inference::boolean_mask;
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Dim, Tensor, TensorType};

/// Select along one or more axes of a tensor with boolean masks, keeping the entries they mark.
///
/// Each masked axis takes a one-dimensional `Bit` mask as long as that axis, given as its own
/// operand after the data. Selected entries keep their relative order and unmasked axes are
/// untouched, so masking axis 1 of a three-dimensional tensor is NumPy's `x[:, mask, :]`.
///
/// Several axes are masked **independently**, making the result the outer product of the per-axis
/// selections: masking axes 0 and 1 is `x[rows][:, cols]`, which keeps a rectangle. Note that this
/// is not NumPy's `x[rows, cols]`, which pairs the two masks up elementwise and returns a
/// one-dimensional result. Independent selection is what this op can type: a mask shaped like the
/// operand would select a ragged region, where each row keeps a different number of entries, and
/// no dense shape can describe that.
///
/// Every masked axis of the result is [`Dim::Bounded`], since the number of entries a mask selects
/// is not known until the mask holds values. This is the op that shot post-selection is built
/// from: mask the shots axis of a [`ShotLoop`](super::ShotLoop) result with a per-shot criterion,
/// and what survives is the shots that passed.
#[derive(Clone)]
pub struct BooleanMask {
    axes: Vec<usize>,
}

impl BooleanMask {
    /// Construct a `BooleanMask` op selecting along each of `axes` independently, taking one mask
    /// operand per axis in the order given.
    ///
    /// The axes must be distinct: masking one axis twice would need the two masks to agree on
    /// which entries survive, which an operand list cannot express.
    pub fn new(axes: impl IntoIterator<Item = usize>) -> Self {
        Self {
            axes: axes.into_iter().collect(),
        }
    }

    /// Return the axes this op selects along, in operand order.
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    /// Split an operand list into the data operand and the one mask per axis that follows it.
    ///
    /// # Panics
    ///
    /// If the list is not [`arity`](ProgramOp::arity) long, which a
    /// [`ProgramFunction`](crate::ProgramFunction) checks as the op is added.
    fn split_operands<'a, T>(&self, operands: &'a [T]) -> (&'a T, &'a [T]) {
        let [x, masks @ ..] = operands else {
            panic!(
                "{} expects {} operand(s), got {}",
                self.full_name(),
                self.arity(),
                operands.len()
            )
        };
        assert_eq!(
            masks.len(),
            self.axes.len(),
            "expected one mask per masked axis"
        );
        (x, masks)
    }

    /// Return the first repeated axis, if this op was given one twice.
    fn repeated_axis(&self) -> Option<usize> {
        let mut seen = Vec::with_capacity(self.axes.len());
        self.axes.iter().copied().find(|axis| {
            let repeated = seen.contains(axis);
            seen.push(*axis);
            repeated
        })
    }
}

impl ProgramOp for BooleanMask {
    type Error = MathOpError;

    fn name(&self) -> &str {
        "boolean_mask"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        // The data, plus one mask per axis.
        self.axes.len() + 1
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        let (x, masks) = self.split_operands(inputs);
        if let Some(axis) = self.repeated_axis() {
            return Err(MathOpError::RepeatedAxis { axis });
        }
        let masks: Vec<(usize, &TensorType)> =
            self.axes.iter().copied().zip(masks.iter()).collect();
        Ok(vec![boolean_mask(x, &masks)?])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        let (x, masks) = self.split_operands(args);
        if let Some(axis) = self.repeated_axis() {
            return Err(MathOpError::RepeatedAxis { axis });
        }
        // Every axis keeps all of its positions unless a mask says otherwise.
        let mut selection: Vec<Vec<usize>> =
            x.shape().iter().map(|&len| (0..len).collect()).collect();
        for (operand, (&axis, mask)) in self.axes.iter().zip(masks).enumerate() {
            check_axis(axis, x.shape().len())?;
            check_dtype(operand + 1, mask.dtype(), DType::Bit)?;
            let Tensor::Bit(bits) = mask else {
                unreachable!("the dtype check admits only a Bit mask")
            };
            let &[length] = bits.shape() else {
                return Err(MathOpError::MaskNotVector { ndim: bits.ndim() });
            };
            // Type inference permits a bounded axis, whose true size only appears here, so this is
            // the first point at which a mask and its axis can actually be compared.
            let selected = x.shape()[axis];
            if length != selected {
                return Err(MathOpError::MaskLength {
                    mask: Dim::Fixed(length),
                    axis,
                    length: Dim::Fixed(selected),
                });
            }
            // Drawn from the axis itself, so every index is in bounds for it.
            selection[axis] = bits
                .iter()
                .enumerate()
                .filter_map(|(index, &bit)| (bit != 0).then_some(index))
                .collect();
        }
        // Selecting the axes independently keeps their outer product, so narrowing them one at a
        // time gives the whole rectangle. Each pass copies into a smaller tensor, and `select_axis`
        // moves contiguous runs of memory rather than visiting elements one by one, so this
        // measures several times faster than a single gather over every axis at once -- the
        // intermediate tensors cost far less than the per-element index arithmetic they avoid.
        let masked = self.axes.iter().fold(x.clone(), |acc, &axis| {
            acc.select_axis(axis, &selection[axis])
        });
        Ok(vec![masked])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::{Parity, ShotLoop};
    use crate::program::ProgramFunction;
    use ndarray::{arr1, arr2, arr3};
    use qiskit_circuit::bit::ClassicalRegister;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::Param;

    /// Return a `TensorType` of `dtype` over `shape`.
    fn ty(dtype: DType, shape: Vec<Dim>) -> TensorType {
        TensorType { dtype, shape }
    }

    /// Return a shape of fixed axes.
    fn fixed(sizes: &[usize]) -> Vec<Dim> {
        sizes.iter().copied().map(Dim::Fixed).collect()
    }

    /// Return the type of a 1-D `Bit` mask of `len` entries.
    fn mask_ty(len: usize) -> TensorType {
        ty(DType::Bit, fixed(&[len]))
    }

    /// Return a 1-D `Bit` tensor holding `bits`.
    fn mask(bits: &[u8]) -> Tensor {
        Tensor::Bit(arr1(bits).into_dyn().into_shared())
    }

    #[test]
    fn test_boolean_mask_full_name_and_arity() {
        let op = BooleanMask::new([0]);
        assert_eq!(op.full_name(), "qiskit.boolean_mask");
        assert_eq!(op.arity(), 2, "the data, plus one mask");
        assert!(op.has_builtin_eval());
        assert_eq!(op.axes(), [0]);

        // Each additional masked axis takes an additional mask operand.
        let many = BooleanMask::new([0, 2]);
        assert_eq!(many.arity(), 3, "the data, plus one mask per axis");
        assert_eq!(many.axes(), [0, 2]);
    }

    #[test]
    fn test_infer_output_types_bounds_the_masked_axis_and_keeps_the_rest() {
        // The masked axis becomes bounded by its own former size; the dtype and other axes pass
        // through untouched.
        assert_eq!(
            BooleanMask::new([0])
                .infer_output_types(&[ty(DType::F64, fixed(&[5, 2])), mask_ty(5)])
                .unwrap(),
            vec![ty(DType::F64, vec![Dim::Bounded { max: 5 }, Dim::Fixed(2)])]
        );
    }

    #[test]
    fn test_infer_output_types_masks_an_axis_other_than_the_first() {
        assert_eq!(
            BooleanMask::new([1])
                .infer_output_types(&[ty(DType::C128, fixed(&[2, 7, 3])), mask_ty(7)])
                .unwrap(),
            vec![ty(
                DType::C128,
                vec![Dim::Fixed(2), Dim::Bounded { max: 7 }, Dim::Fixed(3)]
            )]
        );
    }

    #[test]
    fn test_infer_output_types_takes_the_tighter_of_the_two_bounds() {
        // A bounded axis may be masked: the mask and the axis are equal at run time, so either
        // bound holds and the tighter one is kept.
        let bounded = |max| ty(DType::F64, vec![Dim::Bounded { max }]);
        let bounded_mask = |max| ty(DType::Bit, vec![Dim::Bounded { max }]);

        // A bounded axis with a fixed mask is bounded by the mask.
        assert_eq!(
            BooleanMask::new([0])
                .infer_output_types(&[bounded(100), mask_ty(10)])
                .unwrap(),
            vec![bounded(10)]
        );
        // A fixed axis with a bounded mask is bounded by the axis.
        assert_eq!(
            BooleanMask::new([0])
                .infer_output_types(&[ty(DType::F64, fixed(&[10])), bounded_mask(100)])
                .unwrap(),
            vec![bounded(10)]
        );
        // Two bounded dimensions give the smaller bound.
        assert_eq!(
            BooleanMask::new([0])
                .infer_output_types(&[bounded(100), bounded_mask(40)])
                .unwrap(),
            vec![bounded(40)]
        );
    }

    #[test]
    fn test_infer_output_types_rejects_lengths_that_cannot_match() {
        // Two fixed lengths that differ can never line up.
        assert!(matches!(
            BooleanMask::new([0])
                .infer_output_types(&[ty(DType::F64, fixed(&[5])), mask_ty(4)])
                .unwrap_err(),
            MathOpError::MaskLength {
                mask: Dim::Fixed(4),
                axis: 0,
                length: Dim::Fixed(5)
            }
        ));
        // A fixed length above a bound can never be reached either.
        assert!(matches!(
            BooleanMask::new([0])
                .infer_output_types(&[ty(DType::F64, vec![Dim::Bounded { max: 3 }]), mask_ty(4)])
                .unwrap_err(),
            MathOpError::MaskLength { .. }
        ));
    }

    #[test]
    fn test_infer_output_types_rejects_a_mask_that_is_not_a_bit_vector() {
        // A non-Bit mask must go through an explicit `Cast`.
        assert!(matches!(
            BooleanMask::new([0])
                .infer_output_types(&[ty(DType::F64, fixed(&[3])), ty(DType::U8, fixed(&[3]))])
                .unwrap_err(),
            MathOpError::WrongDType {
                operand: 1,
                expected: DType::Bit,
                actual: DType::U8
            }
        ));
        // A mask must be one-dimensional.
        assert!(matches!(
            BooleanMask::new([0])
                .infer_output_types(&[ty(DType::F64, fixed(&[3])), ty(DType::Bit, fixed(&[3, 1]))])
                .unwrap_err(),
            MathOpError::MaskNotVector { ndim: 2 }
        ));
    }

    #[test]
    fn test_infer_output_types_rejects_an_axis_out_of_bounds() {
        assert!(matches!(
            BooleanMask::new([2])
                .infer_output_types(&[ty(DType::F64, fixed(&[5, 2])), mask_ty(5)])
                .unwrap_err(),
            MathOpError::InvalidAxis { axis: 2, ndim: 2 }
        ));
    }

    #[test]
    fn test_eval_keeps_the_marked_entries_in_order() {
        let x = Tensor::from([10.0_f64, 20.0, 30.0, 40.0]);
        assert_eq!(
            BooleanMask::new([0])
                .eval(&[x, mask(&[1, 0, 0, 1])])
                .unwrap(),
            vec![Tensor::from([10.0_f64, 40.0])]
        );
    }

    #[test]
    fn test_eval_selects_rows_along_the_chosen_axis() {
        let x = Tensor::F64(
            arr2(&[[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]])
                .into_dyn()
                .into_shared(),
        );
        // Masking axis 0 drops a row and keeps both columns.
        assert_eq!(
            BooleanMask::new([0])
                .eval(&[x.clone(), mask(&[1, 0, 1])])
                .unwrap(),
            vec![Tensor::F64(
                arr2(&[[1.0_f64, 2.0], [5.0, 6.0]]).into_dyn().into_shared()
            )]
        );
        // Masking axis 1 drops a column and keeps every row.
        assert_eq!(
            BooleanMask::new([1]).eval(&[x, mask(&[0, 1])]).unwrap(),
            vec![Tensor::F64(
                arr2(&[[2.0_f64], [4.0], [6.0]]).into_dyn().into_shared()
            )]
        );
    }

    #[test]
    fn test_eval_masks_a_middle_axis_of_a_three_dimensional_tensor() {
        let x = Tensor::F64(
            arr3(&[[[1.0_f64], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
                .into_dyn()
                .into_shared(),
        );
        assert_eq!(
            BooleanMask::new([1]).eval(&[x, mask(&[0, 1, 1])]).unwrap(),
            vec![Tensor::F64(
                arr3(&[[[2.0_f64], [3.0]], [[5.0], [6.0]]])
                    .into_dyn()
                    .into_shared()
            )]
        );
    }

    #[test]
    fn test_eval_of_an_all_false_mask_empties_the_axis() {
        let x = Tensor::from([1.0_f64, 2.0, 3.0]);
        let out = BooleanMask::new([0]).eval(&[x, mask(&[0, 0, 0])]).unwrap();
        assert_eq!(out[0].shape(), &[0]);
        assert_eq!(out[0].dtype(), DType::F64);
    }

    #[test]
    fn test_eval_of_an_all_true_mask_returns_the_operand() {
        let x = Tensor::from([1.0_f64, 2.0, 3.0]);
        assert_eq!(
            BooleanMask::new([0])
                .eval(&[x.clone(), mask(&[1, 1, 1])])
                .unwrap(),
            vec![x]
        );
    }

    #[test]
    fn test_eval_rejects_a_mask_whose_length_is_not_the_axis_length() {
        // The check type inference could not make, because a bounded axis is admitted.
        assert!(matches!(
            BooleanMask::new([0])
                .eval(&[Tensor::from([1.0_f64, 2.0, 3.0]), mask(&[1, 0])])
                .unwrap_err(),
            MathOpError::MaskLength {
                mask: Dim::Fixed(2),
                axis: 0,
                length: Dim::Fixed(3)
            }
        ));
    }

    #[test]
    fn test_eval_result_satisfies_the_inferred_type() {
        // The invariant the bounded output axis exists to satisfy: whatever the mask selects, the
        // concrete result still matches the type inferred before it ran.
        let op = BooleanMask::new([0]);
        let inferred = op
            .infer_output_types(&[ty(DType::F64, fixed(&[4, 2])), mask_ty(4)])
            .unwrap();
        let x = Tensor::F64(
            arr2(&[[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
                .into_dyn()
                .into_shared(),
        );
        for bits in [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]] {
            let out = op.eval(&[x.clone(), mask(&bits)]).unwrap();
            assert!(
                out[0].matches(&inferred[0]),
                "result {:?} does not match inferred type {:?} for mask {bits:?}",
                out[0].tensor_type(),
                inferred[0]
            );
        }
    }

    // Masking several axes at once.

    #[test]
    fn test_infer_output_types_bounds_every_masked_axis() {
        // Two masks, two bounded axes; the unmasked one passes straight through.
        assert_eq!(
            BooleanMask::new([0, 2])
                .infer_output_types(&[ty(DType::F64, fixed(&[4, 3, 5])), mask_ty(4), mask_ty(5),])
                .unwrap(),
            vec![ty(
                DType::F64,
                vec![
                    Dim::Bounded { max: 4 },
                    Dim::Fixed(3),
                    Dim::Bounded { max: 5 }
                ]
            )]
        );
    }

    #[test]
    fn test_infer_output_types_names_the_offending_mask_operand() {
        // Operand 0 is the data, so the second mask is operand 2.
        assert!(matches!(
            BooleanMask::new([0, 1])
                .infer_output_types(&[
                    ty(DType::F64, fixed(&[2, 2])),
                    mask_ty(2),
                    ty(DType::U8, fixed(&[2])),
                ])
                .unwrap_err(),
            MathOpError::WrongDType {
                operand: 2,
                expected: DType::Bit,
                actual: DType::U8
            }
        ));
    }

    #[test]
    fn test_masking_one_axis_twice_is_rejected() {
        // Two masks on one axis would have to agree on which entries survive, which the operand
        // list cannot express.
        let operands = [ty(DType::F64, fixed(&[4, 4])), mask_ty(4), mask_ty(4)];
        assert!(matches!(
            BooleanMask::new([1, 1])
                .infer_output_types(&operands)
                .unwrap_err(),
            MathOpError::RepeatedAxis { axis: 1 }
        ));
        assert!(matches!(
            BooleanMask::new([1, 1])
                .eval(&[
                    Tensor::F64(
                        arr2([[1.0_f64, 2.0, 3.0, 4.0]; 4].as_slice())
                            .into_dyn()
                            .into_shared()
                    ),
                    mask(&[1, 0, 1, 0]),
                    mask(&[1, 0, 1, 0]),
                ])
                .unwrap_err(),
            MathOpError::RepeatedAxis { axis: 1 }
        ));
    }

    #[test]
    fn test_eval_keeps_the_rectangle_where_both_masks_agree() {
        // Masking rows and columns independently keeps their outer product: the entries whose row
        // *and* column both survive.
        let x = Tensor::F64(
            arr2(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
                .into_dyn()
                .into_shared(),
        );
        assert_eq!(
            BooleanMask::new([0, 1])
                .eval(&[x, mask(&[1, 0, 1]), mask(&[0, 1, 1])])
                .unwrap(),
            vec![Tensor::F64(
                arr2(&[[2.0_f64, 3.0], [8.0, 9.0]]).into_dyn().into_shared()
            )],
            "rows 0 and 2 crossed with columns 1 and 2"
        );
    }

    #[test]
    fn test_eval_of_several_axes_matches_masking_them_one_at_a_time() {
        // Masking two axes in one op must mean the same thing as masking them with two ops
        let x = Tensor::F64(
            arr3(&[
                [[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ])
            .into_dyn()
            .into_shared(),
        );
        let (rows, cols) = (mask(&[0, 1]), mask(&[1, 0, 1]));

        let at_once = BooleanMask::new([0, 1])
            .eval(&[x.clone(), rows.clone(), cols.clone()])
            .unwrap();

        let first = BooleanMask::new([0]).eval(&[x, rows]).unwrap().remove(0);
        let chained = BooleanMask::new([1]).eval(&[first, cols]).unwrap();

        assert_eq!(at_once, chained);
        assert_eq!(at_once[0].shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_eval_of_several_axes_satisfies_the_inferred_type() {
        let op = BooleanMask::new([0, 1]);
        let operands = [ty(DType::F64, fixed(&[3, 3])), mask_ty(3), mask_ty(3)];
        let inferred = op.infer_output_types(&operands).unwrap();
        let x = Tensor::F64(
            arr2(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
                .into_dyn()
                .into_shared(),
        );
        for rows in [[0, 0, 0], [1, 0, 1], [1, 1, 1]] {
            for cols in [[0, 0, 0], [0, 1, 0], [1, 1, 1]] {
                let out = op.eval(&[x.clone(), mask(&rows), mask(&cols)]).unwrap();
                assert!(
                    out[0].matches(&inferred[0]),
                    "{:?} does not satisfy {:?} for rows {rows:?} and columns {cols:?}",
                    out[0].tensor_type(),
                    inferred[0]
                );
            }
        }
    }

    #[test]
    fn test_eval_of_several_axes_can_empty_the_result() {
        let x = Tensor::F64(arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]).into_dyn().into_shared());
        let out = BooleanMask::new([0, 1])
            .eval(&[x, mask(&[1, 1]), mask(&[0, 0])])
            .unwrap();
        assert_eq!(out[0].shape(), &[2, 0], "no column survives");
        assert_eq!(out[0].dtype(), DType::F64);
    }

    // Shot post-selection, the reason this op exists.

    /// Return a circuit with one classical register of `width` bits and no parameters.
    fn circuit(width: u32) -> CircuitData {
        let mut circuit = CircuitData::new(None, None, Param::Float(0.0)).unwrap();
        circuit
            .add_creg(ClassicalRegister::new_owning("c", width), true)
            .unwrap();
        circuit
    }

    #[test]
    fn test_post_selecting_shot_loop_outcomes_type_checks() {
        // The pipeline a post-selection program is: run the circuit, reduce each shot's bits to a
        // criterion, then keep the shots the criterion marks.
        let mut function = ProgramFunction::new();
        let no_parameters = function.add_parameter(ty(DType::F64, fixed(&[0])));
        let shots = function
            .add_op(
                ShotLoop::new(vec![circuit(2)], 100).unwrap(),
                &[no_parameters],
            )
            .unwrap()[0];
        // Bit[100, 2] -> Bit[100], one criterion per shot.
        let keep = function.add_op(Parity::new(1), &[shots]).unwrap()[0];
        let selected = function
            .add_op(BooleanMask::new([0]), &[shots, keep])
            .unwrap()[0];

        assert_eq!(
            function.type_of(shots),
            Some(&ty(DType::Bit, fixed(&[100, 2])))
        );
        assert_eq!(function.type_of(keep), Some(&ty(DType::Bit, fixed(&[100]))));
        // The shots axis is now bounded: at most 100 survive, and how many is a run-time fact.
        assert_eq!(
            function.type_of(selected),
            Some(&ty(
                DType::Bit,
                vec![Dim::Bounded { max: 100 }, Dim::Fixed(2)]
            ))
        );
    }

    #[test]
    fn test_post_selection_evaluates_through_a_program_function() {
        // A shot loop has no in-process implementation, so the outcomes enter as a parameter here.
        // Everything downstream of them is the same graph as the test above.
        let mut function = ProgramFunction::new();
        let shots = function.add_parameter(ty(DType::Bit, fixed(&[4, 2])));
        let keep = function.add_op(Parity::new(1), &[shots]).unwrap()[0];
        let selected = function
            .add_op(BooleanMask::new([0]), &[shots, keep])
            .unwrap()[0];
        function.add_result(selected).unwrap();

        // Rows of odd parity are kept: [0,1] and [1,0]. The even-parity rows are dropped.
        let outcomes = Tensor::Bit(
            arr2(&[[0_u8, 0], [0, 1], [1, 1], [1, 0]])
                .into_dyn()
                .into_shared(),
        );
        let results = function.eval(&[outcomes]).unwrap();
        assert_eq!(
            results,
            vec![Tensor::Bit(
                arr2(&[[0_u8, 1], [1, 0]]).into_dyn().into_shared()
            )]
        );
    }

    #[test]
    fn test_post_selection_result_count_varies_with_the_data() {
        // The point of a bounded axis: the same function returns a different number of shots for
        // different inputs, and every one of them satisfies the declared output type.
        let mut function = ProgramFunction::new();
        let shots = function.add_parameter(ty(DType::Bit, fixed(&[4, 2])));
        let keep = function.add_op(Parity::new(1), &[shots]).unwrap()[0];
        let selected = function
            .add_op(BooleanMask::new([0]), &[shots, keep])
            .unwrap()[0];
        function.add_result(selected).unwrap();

        // No odd-parity row survives; every row does; and two of four do.
        for (rows, kept) in [
            (vec![[0_u8, 0], [1, 1], [0, 0], [1, 1]], 0),
            (vec![[0_u8, 1], [1, 0], [0, 1], [1, 0]], 4),
            (vec![[0_u8, 0], [1, 0], [1, 1], [0, 1]], 2),
        ] {
            let flat: Vec<u8> = rows.iter().flatten().copied().collect();
            let outcomes = Tensor::Bit(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4, 2]), flat)
                    .unwrap()
                    .into_shared(),
            );
            let results = function.eval(&[outcomes]).unwrap();
            assert_eq!(results[0].shape(), &[kept, 2], "for rows {rows:?}");
        }
    }
}
