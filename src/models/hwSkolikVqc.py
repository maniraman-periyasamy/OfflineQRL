""" Adaptation of a VQC architecture based on Skolik et al. 2022.
    Each layer encodes not only the current observation, but als n past ones.
    https://arxiv.org/pdf/2103.15084.pdf
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vqcBase import VqcBase

class HwSkolikVqc(VqcBase):
    def __init__(self, input_dim, output_dim, n_past_obs=1, num_layers = 1, config=None) -> None:
        # Construct circuit
        # Number of qubits corresponds to input dim.
        self.n_qubits = input_dim
        self.n_layers = num_layers
        circuit = QuantumCircuit(self.n_qubits)

        # If re-uploading adapt architecture
        data_reuploading = True if config["data_reuploading_layers"] > 0 else False

        if data_reuploading:
            input_params = ParameterVector("x", length = (self.n_qubits + n_past_obs * self.n_qubits) * self.n_layers)
            circuit_params = ParameterVector("psi", length = 2* (self.n_qubits + n_past_obs * self.n_qubits) * self.n_layers)

            for layer_idx in range(self.n_layers):
                for obs_idx in range(n_past_obs + 1):
                    for qbit_idx in range(self.n_qubits):
                        circuit.rx(input_params[qbit_idx + (self.n_qubits * obs_idx) + (layer_idx * self.n_qubits * (n_past_obs + 1))], qbit_idx)
                    circuit.barrier()

                    for qbit_idx in range(self.n_qubits):
                        circuit.ry(circuit_params[qbit_idx + 2 * (self.n_qubits * obs_idx + (layer_idx * self.n_qubits * (n_past_obs + 1)))], qbit_idx)
                        circuit.rz(circuit_params[qbit_idx + 2 * (self.n_qubits * obs_idx + (layer_idx * self.n_qubits * (n_past_obs + 1))) + self.n_qubits], qbit_idx)

                    for qbit_idx in range(self.n_qubits-1):
                        circuit.cz(qbit_idx, (qbit_idx+1))
                    circuit.barrier()

        else:
            input_params = ParameterVector("x", length = (self.n_qubits + n_past_obs * self.n_qubits))
            circuit_params = ParameterVector("psi", length = 2* (self.n_qubits + n_past_obs * self.n_qubits))
            # No re-uplaoding just one layer.
            for layer_idx in range(1):
                for obs_idx in range(n_past_obs + 1):
                    for qbit_idx in range(self.n_qubits):
                        circuit.rx(input_params[qbit_idx + (self.n_qubits * obs_idx) + (layer_idx * self.n_qubits * (n_past_obs + 1))], qbit_idx)
                    circuit.barrier()

                    for qbit_idx in range(self.n_qubits):
                        circuit.ry(circuit_params[qbit_idx + 2 * (self.n_qubits * obs_idx + (layer_idx * self.n_qubits * (n_past_obs + 1)))], qbit_idx)
                        circuit.rz(circuit_params[qbit_idx + 2 * (self.n_qubits * obs_idx + (layer_idx * self.n_qubits * (n_past_obs + 1))) + self.n_qubits], qbit_idx)

                    for qbit_idx in range(self.n_qubits-1):
                        circuit.cz(qbit_idx, (qbit_idx+1))
                    circuit.barrier()

        # Construct observable
        observables = [SparsePauliOp(Pauli("ZIIII")), SparsePauliOp(Pauli("IZIII")), SparsePauliOp(Pauli("IIZII")), SparsePauliOp(Pauli("IIIZI")), SparsePauliOp(Pauli("IIIIZ"))]

        # Call parent constructor
        super().__init__(input_dim, input_params, circuit, circuit_params, observables, output_dim, config)
    
    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, model_config:dict):
        return cls( input_dim = model_config["input_dim"],
                    output_dim = model_config["output_dim"],
                    n_past_obs = model_config["n_past_obs"],
                    num_layers = model_config["data_reuploading_layers"],
                    config = model_config
                  )

    def load_pretrained(self, filename:str, load_index_dict:dict):
        """ Load pre-trained weights from a torch checkpoint.
        Args
        filename            Path to the checkpoint.pth
        load_index_dict Dictionary with parameter name as key and index of parameter as value.
        """

        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]

        for name, param in model_state_dict.items():
            param_idxs = load_index_dict.get(name, None)
            if param_idxs is not None:
                if param_idxs == "all":
                    # First two chars in name are either Q. or I.
                    self.state_dict()[name[2:]] = param
                    print(f"Parameter {name} was loaded with all indices.")
                elif type(param_idxs) == list:
                    # First two chars in name are either Q. or I.
                    # Only load parameters at given indices.
                    self.state_dict()[name[2:]][param_idxs] = param[param_idxs]
                    print(f"Parameter {name} was loaded with indices {param_idxs}.")
                else:
                    raise TypeError(f"Type {type(param_idxs)} with vaule {param_idxs} not supported.")