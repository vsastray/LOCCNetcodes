import numpy as np
from numpy import pi as PI
import paddle.fluid as fluid
from paddle.complex import matmul, trace, elementwise_sub
from paddle_quantum.locc import LoccNet
from paddle_quantum.utils import dagger
from utils import ad_state, ad_choi_state


class NetTrainer(LoccNet):
    def __init__(self, gamma):
        super(NetTrainer, self).__init__()
        self.gamma = gamma
        # Add a party, namely Alice, who has two qubits
        self.add_new_party(2, party_name="Alice")
        # Add a party, namely Bob, who has one qubit
        self.add_new_party(1, party_name="Bob")
        # Alice's parameters
        self.theta_a = self.create_parameter(shape=[15], attr=fluid.initializer.Uniform(low=0.0, high=2 * PI),
                                             dtype="float64")
        # Bob's parameters
        self.theta_b = self.create_parameter(shape=[4, 3], attr=fluid.initializer.Uniform(low=0.0, high=2 * PI),
                                             dtype="float64")
        # Initialize quantum state
        _state = fluid.dygraph.to_variable(ad_choi_state(self.gamma))
        # Four linearly independent matrices form a training set
        _state0 = fluid.dygraph.to_variable(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128))
        _state1 = fluid.dygraph.to_variable(np.array([[0, 0], [0, 1]], dtype=np.complex128))
        _state2 = fluid.dygraph.to_variable(np.array([[1, 0], [0, 0]], dtype=np.complex128))
        _state3 = fluid.dygraph.to_variable(np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128))
        self.init_states = [_state0, _state1, _state2, _state3]
        # Set initial state
        self.set_init_status(_state, [("Alice", 0), ("Bob", 0)])
        self.set_init_status(_state0, ("Alice", 1))

    def forward(self):
        loss = 0
        temp_state = self.init_status
        for init_state in self.init_states:
            # Set the state to be teleported
            status = self.reset_state(temp_state, init_state, ("Alice", 1))
            # Alice's circuit
            cir_a = self.create_ansatz("Alice")
            cir_a.universal_2_qubit_gate(self.theta_a, [0, 1])
            status = cir_a.run(status)
            # Measure Alice's qubits
            status_a = self.measure(status, [("Alice", 0), ("Alice", 1)], ["00", "01", "10", "11"])
            # The expected state of Bob's qubit
            tar_state = fluid.dygraph.to_variable(ad_state(self.gamma, init_state.numpy()))
            for idx, s in enumerate(status_a):
                # Bob's circuit
                cir_b = self.create_ansatz("Bob")
                cir_b.u3(*self.theta_b[idx], 0)
                status_b = cir_b.run(s)
                # Obtain the state of Bob's qubit
                status_fin = self.partial_state(status_b, ("Bob", 0))
                # Here we use trace distance as the loss function
                diff = elementwise_sub(status_fin.state, tar_state)
                loss += trace(matmul(diff, dagger(diff))).real
        return loss

    def save_module(self, filename):
        # Save parameters of LOCCNet
        np.savez(filename, theta_a=self.theta_a.numpy(), theta_b=self.theta_b.numpy())


def train(gamma, filename, ITR=200, LR=0.2):
    # Enable the mode of dynamic graph
    with fluid.dygraph.guard():
        net = NetTrainer(gamma)
        opt = fluid.optimizer.AdamOptimizer(learning_rate=LR, parameter_list=net.parameters())
        # Train LOCCNet by gradient descent
        for itr in range(ITR):
            loss = net()
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()
            if itr % 10 == 0:
                print("itr " + str(itr) + ":", loss.numpy()[0])
        net.save_module(filename)
