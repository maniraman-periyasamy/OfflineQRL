from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.algorithms.gradients import SPSAEstimatorGradient
from qiskit_aer.primitives import Estimator, Sampler

# Type setting imports
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit import Parameter as qiskitParameter
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

import numpy as np
import multiprocessing as mp
import copy

from typing import (
    Any,
    Dict,
    Union,
    Optional,
    Sequence
)


# Custom modules
from ..circuitWrapper.base_executer import base_executer
from ..customGradiets.spsa_sampler_gradient import SPSASamplerGradient


class shot_based_executer(base_executer):
    """This class implements the executer using aer simulator.

        Args:
            qc (quantumcircuit): Variational Quantum ansatz
            weight_params (Union[ ParameterVector, Sequence[qiskitParameter]]): List of weight parameters in the VQC
            input_params (Union[ParameterVector, list[qiskitParameter]]): List of input parameters in the VQC
            batch_size (int): Batch size for parallelization and SPSA
            observables (Optional[Sequence[BaseOperator  |  PauliSumOp]], optional): List of quantum observables. Defaults to None.
            grad_type ({"SPSA, "Parameter-shift"}, optional): Gradient estimation method to be used. Defaults to "SPSA".
            epsilon (Optional[float], optional): Epsilon for SPSA. This will be ignored in case of parameter-shift type gradient estimation Defaults to 0.2.
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
            shots (Optional[int], optional): Number of shots to be used for shot based simulator. Defaults to 1024.

    """

    def __init__(
            self,  qc: QuantumCircuit,
            weight_params: Union[ParameterVector, Sequence[qiskitParameter]],
            input_params: Union[ParameterVector, list[qiskitParameter]],
            batch_size: int,
            observables: Optional[Sequence[BaseOperator | PauliSumOp]] = None,
            grad_type: Optional[str] = "SPSA", epsilon: Optional[float] = 0.2,
            num_parallel_process: Optional[int] = None,
            shots: Optional[int] = 1024
            ) -> None:

        super(shot_based_executer, self).__init__(
            qc=qc, weight_params=weight_params, input_params=input_params,
            batch_size=batch_size,
            observables=observables, grad_type=grad_type, epsilon=epsilon,
            num_parallel_process=num_parallel_process, shots=shots)
        self.shots = shots

    def set_executer(self, session=None, options=None, noise_model=None, coupling_map=None, basis_gates=None):
        """This function sets the executer for the vqc and the respective gradient estimation method.

        Args:
            session (QiskitRuntime Session, optional): Qiskit runtime session to use. Defaults to None.
            options (QiskitRuntime Session Options, optional): Transpilation and error mitigation options. If None, highest possible setting is used. Defaults to None.
        """

        if self.grad_type == "SPSA":
            if self.observables is None:
                self.executer = Sampler(run_options={"shots": self.shots}, backend_options={
                    "method": "density_matrix",
                    "coupling_map": coupling_map,
                    "noise_model": noise_model,
                    "basis_gates": basis_gates
                })
                self.Grad_executer = SPSASamplerGradient(
                                                        self.executer,
                                                        epsilon=self.epsilon,
                                                        batch_size=self.batch_size)
            else:
                self.executer = Estimator(run_options={"shots": self.shots}, backend_options={
                    "method": "density_matrix",
                    "coupling_map": coupling_map,
                    "noise_model": noise_model,
                    "basis_gates": basis_gates
                })
                self.Grad_executer = SPSAEstimatorGradient(
                                                            self.executer,
                                                            epsilon=self.epsilon,
                                                            batch_size=self.batch_size)
        else:
            if self.observables is None:
                self.executer = Sampler(run_options={"shots": self.shots}, backend_options={
                    "method": "density_matrix",
                    "coupling_map": coupling_map,
                    "noise_model": noise_model,
                    "basis_gates": basis_gates
                })
                self.Grad_executer = ParamShiftEstimatorGradient(
                                                                self.executer)

            else:
                self.executer = Estimator(run_options={"shots": self.shots}, backend_options={
                    "method": "density_matrix",
                    "coupling_map": coupling_map,
                    "noise_model": noise_model,
                    "basis_gates": basis_gates
                })
                self.Grad_executer = ParamShiftEstimatorGradient(
                                                                self.executer)
