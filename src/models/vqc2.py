from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vqcBase import VqcBase

class Vqc2(VqcBase):
    def __init__(self, input_dim, output_dim, num_layers = 1, parameters=None) -> None:

        self.n_qubits = input_dim
        self.n_layers = num_layers
        circuit = QuantumCircuit(self.n_qubits)
        
        input_params = ParameterVector("x", length=self.n_qubits)
        circuit_params = ParameterVector("psi", length=self.n_qubits*self.n_layers*2)

        # If re-uploading adapt architecture
        data_reuploading = True if parameters["data_reuploading_layers"] > 0 else False
        
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

        observables = [SparsePauliOp(Pauli("XIII")), SparsePauliOp(Pauli("IYII")), SparsePauliOp(Pauli("IIZI")),
                       SparsePauliOp(Pauli("IIIX")), SparsePauliOp(Pauli("XYII")), SparsePauliOp(Pauli("IYZI")),
                       SparsePauliOp(Pauli("IIZX")), SparsePauliOp(Pauli("XYZI")), SparsePauliOp(Pauli("IYZX"))]

        # Call parent constructor
        super().__init__(input_dim, input_params, circuit, circuit_params, observables, output_dim, parameters)
"""
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        batch = x.shape[0]
        logits = self.net(x.view(batch, -1))

        # Return logits and softmax
        return logits, F.log_softmax(logits, dim=1)"""
    