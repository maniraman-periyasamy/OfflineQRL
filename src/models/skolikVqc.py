""" Implementation of a VQC architecture based on Skolik et al. 2022.
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

class SkolikVqc(VqcBase):
    def __init__(self, input_dim, output_dim, num_layers = 1, config=None) -> None:
        # Construct circuit
        self.n_qubits = input_dim
        self.n_layers = num_layers
        circuit = QuantumCircuit(self.n_qubits)
        
        input_params = ParameterVector("x", length=self.n_qubits)
        circuit_params = ParameterVector("psi", length=self.n_qubits*self.n_layers*2)

        # If re-uploading adapt architecture
        data_reuploading = True if config["data_reuploading_layers"] > 0 else False

        if data_reuploading:
            input_params = ParameterVector("x", length=self.n_qubits*self.n_layers)
        else:
            for i in range(self.n_qubits):
                circuit.rx(input_params[i], i)
            circuit.barrier()

        # Construct the variational layers
        for i in range(self.n_layers):
            if data_reuploading:
                for k in range(self.n_qubits):
                    circuit.rx(input_params[i*self.n_qubits + k], k)
                circuit.barrier()

            for j in range(self.n_qubits):
                circuit.ry(circuit_params[2*i*self.n_qubits + j], j)
                circuit.rz(circuit_params[(2*i+1)*self.n_qubits + j], j)

            for j in range(self.n_qubits-1):
                circuit.cz(j, (j+1))
            circuit.barrier()

        # Construct observable
        observables = [SparsePauliOp(Pauli("ZZII")), SparsePauliOp(Pauli("IIZZ"))]

        # Call parent constructor
        super().__init__(input_dim, input_params, circuit, circuit_params, observables, output_dim, config)
    
    @classmethod
    # Factory method similar to constructor overloading
    def fromDict(cls, model_config:dict):
        return cls( input_dim = model_config["input_dim"],
                    output_dim = model_config["output_dim"],
                    num_layers = model_config["data_reuploading_layers"],#model_params["num_layers"],
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