import torch
from torch.nn import (Module, Parameter)

# Type setting imports
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit import Parameter as qiskitParameter
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


import numpy as np
import copy
from typing import (
    Any,
    Dict,
    Union,
    Optional,
    Sequence
)


# Custom modules
from ..circuitWrapper.analytical_executer import analytical_executer
from ..circuitWrapper.shot_based_executer import shot_based_executer
from ..circuitWrapper.ibmq_executer import ibmq_executer
from ..quantumLayer.torchFunction import HybridFunction


class add_weights(Module):
    """This class implements the fuctionality of adding a fully connected layer or a scaling parameter to the inputs for and/or outputs from VQC.

        Args:
            input_size (Optional[int], optional): Input dimension. Defaults to 0.
            weights_type ({"unique", "same"}, optional): Flag to indicate weights. Defaults to 'unique'.
            sampling_type ({"random", "ones"}, optional): Type of weight initialization. Defaults to 'random'.
            name (Optional[str], optional): Name of this layer. Defaults to 'weight_layer'.
            process_input ({"None", "skolik"}, optional): Postprocessing method to be applied. Defaults to "None".
    """

    def __init__(
                self,
                input_size: Optional[int] = 0,
                weights_type: Optional[str] = 'unique',
                sampling_type: Optional[str] = 'random',
                name: Optional[str] = 'weight_layer',
                process_input: Optional[str] = "None") -> None:

        super().__init__()

        if weights_type == 'unique':
            self.input_size = (1, input_size)
        else:
            self.input_size = 1

        if sampling_type == 'random':
            self.trainable_parameters = torch.tensor(
                np.random.uniform(low=0.0, high=np.pi, size=self.input_size).astype(np.float32),
                device=torch.device("cpu"), requires_grad=True,
                dtype=torch.float32)
        elif sampling_type == 'ones':
            self.trainable_parameters = torch.tensor(
                np.ones(self.input_size).astype(np.float32),
                device=torch.device("cpu"),
                requires_grad=True,
                dtype=torch.float32)
        self.process_input = process_input
        self.trainable_parameters = Parameter(self.trainable_parameters)

    def forward(self, inputs):
        if self.process_input == "None":
            output = inputs*self.trainable_parameters
        elif self.process_input == 'skolik':
            inputs = (inputs+1)/2
            output = inputs*self.trainable_parameters
        return output



class data_reuploading(Module):
    
    def __init__(self, num_layers = 4, method='normal', name='data_reuploading_layer'):
        """This layer tiles the input to enable data-reuploading functionality.

        Args:
            num_layers (int, optional): Number of repetitions of the input vector . Defaults to 4.
            method (str, optional): Type of data-reuploading to be implemented.
            name (str, optional): Custom name for the layer. Defaults to 'data_reuploading_layer'.
        """
        super().__init__()

        self.num_layers = num_layers
        self.method = method

    def forward(self, x):
        if self.method == "normal":
            x =  torch.tile(x, (self.num_layers,))
        elif self.method == "cyclic":
            x_temp = copy.deepcopy(x)
            for i in range(1, self.num_layers):
                x_temp = torch.roll(x_temp, 1, dims=1)
                x = torch.column_stack((x, x_temp))
        
        """if tf.rank(x) == 2:
            x = tf.expand_dims(x, axis=0)"""
        return x

class base_layer(Module):
    """This layer acts as a wrapper that connects the quantum function to the classical functions.

        Args:
            qc (quantumcircuit): Variational Quantum ansatz
            weight_params (Union[ ParameterVector, Sequence[qiskitParameter]]): List of weight parameters in the VQC
            input_params (Union[ParameterVector, list[qiskitParameter]]): List of input parameters in the VQC
            batch_size (int): Batch size for parallelization and SPSA
            data_reuploading_layers (Optional[int], optional): Number of data-reuploading layers. Defaults to 0.
            data_reuploading_type (Optional[str], optional):Type of data-reuploading to use. Defaults to normal.
            observables (Optional[Sequence[BaseOperator  |  PauliSumOp]], optional): List of quantum observables. Defaults to None.
            grad_type ({"SPSA, "Parameter-shift"}, optional): Gradient estimation method to be used. Defaults to "SPSA".
            epsilon (Optional[float], optional): Epsilon for SPSA. This will be ignored in case of parameter-shift type gradient estimation Defaults to 0.2.
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
            quantum_compute_method ({"analytical, "shot-based", "ibmq"}, optional): Type of executer to use. Defaults to "analytical".
            quantum_weight_initialization ({"random, "ones", "zeros"}, optional): Type of weight initialization. Defaults to 'random'.
            output_dim (Optional[int], optional): Number of outputs from VQC. Defaults to 2.
            input_dim (Optional[int], optional): Number of inputs to the VQC. Defaults to 4.
            add_input_weights (Optional[bool], optional): Add a fully connected layer to the input. Defaults to False.
            add_output_weights (Optional[bool], optional): Add a fully connected layer to the output. Defaults to False.
            input_scaling (Optional[bool], optional): Add a scaling parameter to the inputs. Defaults to False.
            output_scaling (Optional[bool], optional): Add a scaling parameter to the outputs. Defaults to True.
            input_weight_initialization ({"random, "ones"}, optional): Type of weight initialization. Defaults to 'random'.
            output_weight_initialization ({"random, "ones"}, optional): Type of weight initialization. Defaults to 'ones'.
            post_process_decode (Optional[str], optional): Ignored for now. Defaults to "None".
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
            shots (Optional[int], optional): Number of shots to be used for shot based simulator. Defaults to 1024.
            ibmq_session_max_time (Optional[int], optional): Maximum time for ibmq session in seconds. Defaults to 25200.
        """

    def __init__(
                self,
                qc: QuantumCircuit,
                weight_params: Union[ParameterVector, Sequence[qiskitParameter]],
                input_params: Union[ParameterVector, list[qiskitParameter]],
                batch_size: int,
                data_reuploading_layers: Optional[int] = 0,
                data_reuploading_type: Optional[str] = "normal",
                observables: Optional[Sequence[BaseOperator | PauliSumOp]] = None,
                grad_type: Optional[str] = "SPSA", epsilon: Optional[float] = 0.2,
                quantum_compute_method: Optional[str] = "analytical",
                quantum_weight_initialization: Optional[str] = 'random',
                output_dim: Optional[int] = 2,
                input_dim: Optional[int] = 4,
                add_input_weights: Optional[bool] = False,
                add_output_weights: Optional[bool] = False,
                input_scaling: Optional[bool] = False,
                output_scaling: Optional[bool] = True,
                input_weight_initialization: Optional[str] = 'random',
                output_weight_initialization: Optional[str] = 'ones',
                post_process_decode: Optional[str] = "None",
                num_parallel_process: Optional[int] = None,
                shots: Optional[int] = 1024,
                ibmq_session_max_time: Optional[int] = 7*60*60):
        super().__init__()

        if quantum_compute_method == "analytical":
            self.QcN = analytical_executer(
                qc=qc, weight_params=weight_params, input_params=input_params,
                batch_size=batch_size, observables=observables,
                grad_type=grad_type, epsilon=epsilon,
                num_parallel_process=num_parallel_process)
        elif quantum_compute_method == "shot-based":
            self.QcN = shot_based_executer(qc=qc, weight_params=weight_params, input_params=input_params,
                batch_size=batch_size, observables=observables,
                grad_type=grad_type, epsilon=epsilon,
                num_parallel_process=num_parallel_process, shots=shots)
        elif quantum_compute_method == "ibmq":
            self.QcN = ibmq_executer(qc=qc, weight_params=weight_params, input_params=input_params,
                batch_size=batch_size, observables=observables,
                grad_type=grad_type, epsilon=epsilon,
                num_parallel_process=num_parallel_process, shots=shots, ibmq_session_max_time=ibmq_session_max_time)

        self.data_reuploading_layers = data_reuploading_layers
        self.add_input_weights = add_input_weights
        self.add_output_weights = add_output_weights
        self.input_scaling = input_scaling
        self.output_scaling = output_scaling
        input_weight_shape = input_dim

        if self.data_reuploading_layers != 0:
            self.data_reLayer = data_reuploading(data_reuploading_layers, method=data_reuploading_type)
            input_weight_shape = input_weight_shape*data_reuploading_layers

        if quantum_weight_initialization == 'random':
            self.trainable_parameters = np.random.uniform(low=0.0, high=np.pi, size=len(self.QcN.weight_parameters)).astype(np.float32)
        elif quantum_weight_initialization == 'zeros':
            self.trainable_parameters = np.zeros(len(self.QcN.weight_parameters)).astype(np.float32)
        elif quantum_weight_initialization == 'ones':
            self.trainable_parameters = np.ones(len(self.QcN.weight_parameters)).astype(np.float32)

        self.trainable_parameters = torch.tensor(
            data=self.trainable_parameters, device=torch.device("cpu"),
            requires_grad=True, dtype=torch.float32)

        self.trainable_parameters = Parameter(self.trainable_parameters)

        if self.add_input_weights:
            self.input_weights_layer = add_weights(
                input_size=input_weight_shape, weights_type='unique',
                sampling_type=input_weight_initialization, name='input_weights')
        if self.input_scaling:
            self.input_scaling_layer = add_weights(
                input_size=1, weights_type='same',
                sampling_type='ones', name='input_scaling')
        if self.add_output_weights:
            self.output_weights_layer = add_weights(
                input_size=output_dim, weights_type='unique',
                sampling_type=output_weight_initialization,
                name='output_weights',
                process_input=post_process_decode)
        if self.output_scaling:
            self.output_scaling_layer = add_weights(
                input_size=1,
                weights_type='same',
                sampling_type='ones',
                name='output_scaling', process_input=post_process_decode)

        self.shift = np.pi/2

    def set_session(self, session=None, options=None, noise_model=None, coupling_map=None):
        self.QcN.set_executer(session=session, options=options, noise_model=noise_model, coupling_map=coupling_map)

    def check_time_up_flag(self):
        return self.QcN.ibmq_session_timeup_flag

class torch_layer(base_layer):
    """This layer generates a quantum torch layer to connect the torch API.

        Args:
            qc (quantumcircuit): Variational Quantum ansatz
            weight_params (Union[ ParameterVector, Sequence[qiskitParameter]]): List of weight parameters in the VQC
            input_params (Union[ParameterVector, list[qiskitParameter]]): List of input parameters in the VQC
            batch_size (int): Batch size for parallelization and SPSA
            data_reuploading_layers (Optional[int], optional): Number of data-reuploading layers. Defaults to 0.
            data_reuploading_type (Optional[str], optional):Type of data-reuploading to use. Defaults to normal.
            observables (Optional[Sequence[BaseOperator  |  PauliSumOp]], optional): List of quantum observables. Defaults to None.
            grad_type ({"SPSA, "Parameter-shift"}, optional): Gradient estimation method to be used. Defaults to "SPSA".
            epsilon (Optional[float], optional): Epsilon for SPSA. This will be ignored in case of parameter-shift type gradient estimation Defaults to 0.2.
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
            quantum_compute_method ({"analytical, "shot-based", "ibmq"}, optional): Type of executer to use. Defaults to "analytical".
            quantum_weight_initialization ({"random, "ones", "zeros"}, optional): Type of weight initialization. Defaults to 'random'.
            output_dim (Optional[int], optional): Number of outputs from VQC. Defaults to 2.
            input_dim (Optional[int], optional): Number of inputs to the VQC. Defaults to 4.
            add_input_weights (Optional[bool], optional): Add a fully connected layer to the input. Defaults to False.
            add_output_weights (Optional[bool], optional): Add a fully connected layer to the output. Defaults to False.
            input_scaling (Optional[bool], optional): Add a scaling parameter to the inputs. Defaults to False.
            output_scaling (Optional[bool], optional): Add a scaling parameter to the outputs. Defaults to True.
            input_weight_initialization ({"random, "ones"}, optional): Type of weight initialization. Defaults to 'random'.
            output_weight_initialization ({"random, "ones"}, optional): Type of weight initialization. Defaults to 'ones'.
            post_process_decode (Optional[str], optional): Ignored for now. Defaults to "None".
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
            shots (Optional[int], optional): Number of shots to be used for shot based simulator. Defaults to 1024.
            ibmq_session_max_time (Optional[int], optional): Maximum time for ibmq session in seconds. Defaults to 25200
            fix_parameter_indices: Indices of VQC parameters that are kept static throught training.
        """
    def __init__(
                self,
                qc: QuantumCircuit,
                weight_params: Union[ParameterVector, Sequence[qiskitParameter]],
                input_params: Union[ParameterVector, list[qiskitParameter]],
                batch_size: int,
                data_reuploading_layers: Optional[int] = 0,
                data_reuploading_type: Optional[str] = "normal",
                observables: Optional[Sequence[BaseOperator | PauliSumOp]] = None,
                grad_type: Optional[str] = "SPSA",
                epsilon: Optional[float] = 0.2,
                quantum_compute_method: Optional[str] = "analytical",
                quantum_weight_initialization: Optional[str] = 'random',
                output_dim: Optional[int] = 2,
                input_dim: Optional[int] = 4,
                add_input_weights: Optional[bool] = False,
                add_output_weights: Optional[bool] = False,
                input_scaling: Optional[bool] = False,
                output_scaling: Optional[bool] = True,
                input_weight_initialization: Optional[str] = 'random',
                output_weight_initialization: Optional[str] = 'ones',
                post_process_decode: Optional[str] = "None",
                num_parallel_process: Optional[int] = None,
                shots: Optional[int] = 1024,
                ibmq_session_max_time: Optional[int] = 7*60*60,
                fix_parameter_indices: Optional[list] = None):

        super().__init__(
            qc, weight_params, input_params, batch_size, data_reuploading_layers, data_reuploading_type,
            observables, grad_type, epsilon, quantum_compute_method,
            quantum_weight_initialization, output_dim, input_dim,
            add_input_weights, add_output_weights, input_scaling,
            output_scaling,
            input_weight_initialization, output_weight_initialization,
            post_process_decode, num_parallel_process=num_parallel_process, shots=shots, ibmq_session_max_time=ibmq_session_max_time)
        
        # The static parameters should be masked in the backward step i.e., gradient is set to.
        self.gradient_mask = torch.ones_like(self.trainable_parameters)
        
        if fix_parameter_indices is not None:
            self.gradient_mask[fix_parameter_indices] = 0.0

    def forward(self, input):

        if not torch.is_tensor(input):
            input = torch.tensor(input, requires_grad=True)

        if self.data_reuploading_layers != 0:
            input = self.data_reLayer(input)

        if self.add_input_weights:
            input = self.input_weights_layer(input)
        if self.input_scaling:
            input = self.input_scaling_layer(input)

        input = HybridFunction.apply(
            self.QcN, input, self.trainable_parameters, self.shift, self.gradient_mask)

        if self.add_output_weights:
            input = self.output_weights_layer(input)
        if self.output_scaling:
            input = self.output_scaling_layer(input)

        return input


class tianshou_layer(base_layer):
    """This layer generates a quantum torch layer to connect the tianshou API.

        Args:
            qc (quantumcircuit): Variational Quantum ansatz
            weight_params (Union[ ParameterVector, Sequence[qiskitParameter]]): List of weight parameters in the VQC
            input_params (Union[ParameterVector, list[qiskitParameter]]): List of input parameters in the VQC
            batch_size (int): Batch size for parallelization and SPSA
            data_reuploading_layers (Optional[int], optional): Number of data-reuploading layers. Defaults to 0.
            data_reuploading_type (Optional[str], optional):Type of data-reuploading to use. Defaults to normal.
            observables (Optional[Sequence[BaseOperator  |  PauliSumOp]], optional): List of quantum observables. Defaults to None.
            grad_type ({"SPSA, "Parameter-shift"}, optional): Gradient estimation method to be used. Defaults to "SPSA".
            epsilon (Optional[float], optional): Epsilon for SPSA. This will be ignored in case of parameter-shift type gradient estimation Defaults to 0.2.
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
            quantum_compute_method ({"analytical, "shot-based", "ibmq"}, optional): Type of executer to use. Defaults to "analytical".
            quantum_weight_initialization ({"random, "ones", "zeros"}, optional): Type of weight initialization. Defaults to 'random'.
            output_dim (Optional[int], optional): Number of outputs from VQC. Defaults to 2.
            input_dim (Optional[int], optional): Number of inputs to the VQC. Defaults to 4.
            add_input_weights (Optional[bool], optional): Add a fully connected layer to the input. Defaults to False.
            add_output_weights (Optional[bool], optional): Add a fully connected layer to the output. Defaults to False.
            input_scaling (Optional[bool], optional): Add a scaling parameter to the inputs. Defaults to False.
            output_scaling (Optional[bool], optional): Add a scaling parameter to the outputs. Defaults to True.
            input_weight_initialization ({"random, "ones"}, optional): Type of weight initialization. Defaults to 'random'.
            output_weight_initialization ({"random, "ones"}, optional): Type of weight initialization. Defaults to 'ones'.
            post_process_decode (Optional[str], optional): Ignored for now. Defaults to "None".
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
            shots (Optional[int], optional): Number of shots to be used for shot based simulator. Defaults to 1024.
            ibmq_session_max_time (Optional[int], optional): Maximum time for ibmq session in seconds. Defaults to 25200
        """

    def __init__(
                self,
                qc: QuantumCircuit,
                weight_params: Union[ParameterVector, Sequence[qiskitParameter]],
                input_params: Union[ParameterVector, list[qiskitParameter]],
                batch_size: int, 
                data_reuploading_layers: Optional[int] = 0,
                data_reuploading_type: Optional[str] = "normal",
                observables: Optional[Sequence[BaseOperator | PauliSumOp]] = None,
                grad_type: Optional[str] = "SPSA",
                epsilon: Optional[float] = 0.2,
                quantum_compute_method: Optional[str] = "analytical",
                quantum_weight_initialization: Optional[str] = 'random',
                output_dim: Optional[int] = 2,
                input_dim: Optional[int] = 4,
                add_input_weights: Optional[bool] = False,
                add_output_weights: Optional[bool] = False,
                input_scaling: Optional[bool] = False,
                output_scaling: Optional[bool] = True,
                input_weight_initialization: Optional[str] = 'random',
                output_weight_initialization: Optional[str] = 'ones',
                post_process_decode: Optional[str] = "None",
                num_parallel_process: Optional[int] = None,
                shots: Optional[int] = 1024,
                ibmq_session_max_time: Optional[int] = 7*60*60):

        super().__init__(
            qc, weight_params, input_params, batch_size, data_reuploading_layers, data_reuploading_type,
            observables, grad_type, epsilon, quantum_compute_method,
            quantum_weight_initialization, output_dim, input_dim,
            add_input_weights, add_output_weights, input_scaling,
            output_scaling,
            input_weight_initialization, output_weight_initialization,
            post_process_decode, num_parallel_process=num_parallel_process, shots=shots, ibmq_session_max_time=ibmq_session_max_time)

    def forward(self, obs, state: Any = None, info: Dict[str, Any] = {}):

        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, requires_grad=True)
        
        if self.data_reuploading_layers != 0:
            obs = self.data_reLayer(obs)
        
        if self.add_input_weights:
            obs = self.input_weights_layer(obs)
        if self.input_scaling:
            obs = self.input_scaling_layer(obs)

        obs = HybridFunction.apply(
            self.QcN, obs, self.trainable_parameters, self.shift)

        if self.add_output_weights:
            obs = self.output_weights_layer(obs)
        if self.output_scaling:
            obs = self.output_scaling_layer(obs)

        return obs, state
