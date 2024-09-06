from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vqcBase import VqcBase

class Vqc3(VqcBase):
    def __init__(self, input_dim, output_dim, num_layers = 1, parameters=None) -> None:

        self.n_qubits = 4
        self.n_layers = num_layers
        circuit = QuantumCircuit(self.n_qubits)
        
        input_params = ParameterVector("x", length=input_dim)
        circuit_params = ParameterVector("psi", length=self.n_qubits*self.n_layers*4)

        # If re-uploading adapt architecture
        data_reuploading = True if parameters["data_reuploading_layers"] > 0 else False
        
        if data_reuploading:
            input_params = ParameterVector("x", length=input_dim*self.n_layers)
        
        # Each layer consists of two encoding layers with subsequent variational layers
        for i in range(0, self.n_layers*2, 2):
            # State space shape: pos, theta, action, deltaPos, deltaTheta
            for qubit, k in enumerate([0,1,3,4]):
                circuit.rx(input_params[k], qubit)

            circuit.cry(input_params[2], 0, 1)
            circuit.cry(input_params[2], 2, 3)  
            circuit.barrier()

            for j in range(self.n_qubits):
                circuit.cx(j, (j+1) % self.n_qubits)
        
            for j in range(self.n_qubits):
                circuit.rz(circuit_params[2*i*self.n_qubits + j], j)
                circuit.ry(circuit_params[(2*i+1)*self.n_qubits + j], j)

            for j in range(self.n_qubits):
                circuit.cx(j, (j+1) % self.n_qubits)
            circuit.barrier()

            for qubit, k in enumerate([5,6,8,9]):
                circuit.rx(input_params[k], qubit)

            circuit.cry(input_params[7], 0, 1)
            circuit.cry(input_params[7], 2, 3)  
            circuit.barrier()

            for j in range(self.n_qubits):
                circuit.cx(j, (j+1) % self.n_qubits)
        
            i += 1
            for j in range(self.n_qubits):
                circuit.rz(circuit_params[2*i*self.n_qubits + j], j)
                circuit.ry(circuit_params[(2*i+1)*self.n_qubits + j], j)

            for j in range(self.n_qubits):
                circuit.cx(j, (j+1) % self.n_qubits)
            circuit.barrier()


        circuit.draw(output="mpl", reverse_bits=True).savefig("test")

        observables = [SparsePauliOp(Pauli("XXXX")), SparsePauliOp(Pauli("YYYY")), SparsePauliOp(Pauli("ZZZZ"))]
        
        # Call parent constructor
        super().__init__(input_dim, input_params, circuit, circuit_params, observables, output_dim, parameters)
    