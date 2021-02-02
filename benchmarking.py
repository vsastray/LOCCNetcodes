import numpy as np
import paddle.fluid as fluid
from paddle_quantum.locc import LoccNet
from paddle_quantum.utils import state_fidelity
from paddle_quantum.state import density_op_random
from utils import ad_state, ad_choi_state


class NetEvaluator(LoccNet):
    def __init__(self, param, gamma):
        super(NetEvaluator, self).__init__()
        self.gamma = gamma
        # Add a party, namely Alice, who has two qubits
        self.add_new_party(2, party_name="Alice")
        # Add a party, namely Bob, who has one qubit
        self.add_new_party(1, party_name="Bob")
        # Alice‘s parameters
        self.theta_a = fluid.dygraph.to_variable(param["theta_a"])
        # Bob's parameters
        self.theta_b = fluid.dygraph.to_variable(param["theta_b"])
        # Prepare AD channel's Choi state
        _state = fluid.dygraph.to_variable(ad_choi_state(self.gamma))
        # Generate a random pure state
        random_state = density_op_random(n=1, real_or_complex=1, rank=1)
        self._state0 = fluid.dygraph.to_variable(random_state)
        # Set initial state
        self.set_init_status(_state, [("Alice", 0), ("Bob", 0)])
        self.set_init_status(self._state0, ("Alice", 1))

    def learned(self):
        # Test state
        input_state = self.init_status
        # Alice's circuit
        cir_a = self.create_ansatz("Alice")
        cir_a.universal_2_qubit_gate(self.theta_a, [0, 1])
        status = cir_a.run(input_state)
        status_a = self.measure(status, [("Alice", 0), ("Alice", 1)], ["00", "01", "10", "11"])
        # The expected state of Bob's qubit
        tar_state = fluid.dygraph.to_variable(ad_state(self.gamma, self._state0.numpy()))
        average_fid = 0
        for idx, s in enumerate(status_a):
            # Bob's circuit
            cir_b = self.create_ansatz("Bob")
            cir_b.u3(*self.theta_b[idx], 0)
            status_b = cir_b.run(s)
            # Obtain the state of Bob's qubit
            status_fin = self.partial_state(status_b, ("Bob", 0))
            # Compute fidelity of the state that Bob receives to the expected state
            fid = state_fidelity(tar_state.numpy(), status_fin.state.numpy()) ** 2
            average_fid += fid * status_fin.prob.numpy()[0]

        return average_fid

    def original(self):
        status = self.init_status
        # Alice's circuit
        cir_a = self.create_ansatz("Alice")
        cir_a.cnot([1, 0])
        cir_a.h(1)
        status = cir_a.run(status)
        # Measure Alice's qubits
        status_a = self.measure(status, [("Alice", 0), ("Alice", 1)], ["00", "01", "10", "11"])
        # The expected state of Bob's qubit
        tar_state = ad_state(self.gamma, self._state0.numpy())
        average_fid = 0
        for idx, s in enumerate(status_a):
            # Bob's circuit
            cir_b = self.create_ansatz("Bob")
            if status_a[idx].measured_result[0] == '1':
                cir_b.x(0)
            if status_a[idx].measured_result[1] == '1':
                cir_b.z(0)
            status_b = cir_b.run(s)
            # Obtain the state of Bob's qubit
            status_fin = self.partial_state(status_b, ("Bob", 0))
            # Compute fidelity of the state that Bob receives to the expected state
            fid = state_fidelity(tar_state, status_fin.state.numpy()) ** 2
            average_fid += fid * status_fin.prob.numpy()[0]

        return average_fid


def benchmark(gamma, filename, samples=1000):
    with fluid.dygraph.guard():
        # Load parameters of LOCCNet
        param = np.load(filename)
        list_loccnet = []
        list_teleport = []
        for i in range(samples):
            net = NetEvaluator(param, gamma)
            list_loccnet.append(net.learned())
            list_teleport.append(net.original())
        # Print benchmark results
        print("learned_average:", sum(list_loccnet) / len(list_loccnet), "std=", np.std(list_loccnet))
        print("original_average:", sum(list_teleport) / len(list_teleport), "std=", np.std(list_teleport))
    return list_loccnet, list_teleport
